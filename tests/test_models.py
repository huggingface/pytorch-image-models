import pytest
import torch
import platform
import os
import fnmatch

from timm import list_models, create_model, set_scriptable


if 'GITHUB_ACTIONS' in os.environ and 'Linux' in platform.system():
    # GitHub Linux runner is slower and hits memory limits sooner than MacOS, exclude bigger models
    EXCLUDE_FILTERS = ['*efficientnet_l2*', '*resnext101_32x48d']
else:
    EXCLUDE_FILTERS = []
MAX_FWD_SIZE = 384
MAX_BWD_SIZE = 128
MAX_FWD_FEAT_SIZE = 448


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=EXCLUDE_FILTERS))
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False)
    model.eval()

    input_size = model.default_cfg['input_size']
    if any([x > MAX_FWD_SIZE for x in input_size]):
        # cap forward test at max res 448 * 448 to keep resource down
        input_size = tuple([min(x, MAX_FWD_SIZE) for x in input_size])
    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)

    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=EXCLUDE_FILTERS))
@pytest.mark.parametrize('batch_size', [2])
def test_model_backward(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False, num_classes=42)
    num_params = sum([x.numel() for x in model.parameters()])
    model.eval()

    input_size = model.default_cfg['input_size']
    if any([x > MAX_BWD_SIZE for x in input_size]):
        # cap backward test at 128 * 128 to keep resource usage down
        input_size = tuple([min(x, MAX_BWD_SIZE) for x in input_size])
    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)
    outputs.mean().backward()
    for n, x in model.named_parameters():
        assert x.grad is not None, f'No gradient for {n}'
    num_grad = sum([x.grad.numel() for x in model.parameters() if x.grad is not None])

    assert outputs.shape[-1] == 42
    assert num_params == num_grad, 'Some parameters are missing gradients'
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models())
@pytest.mark.parametrize('batch_size', [1])
def test_model_default_cfgs(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False)
    model.eval()
    state_dict = model.state_dict()
    cfg = model.default_cfg

    classifier = cfg['classifier']
    first_conv = cfg['first_conv']
    pool_size = cfg['pool_size']
    input_size = model.default_cfg['input_size']

    if all([x <= MAX_FWD_FEAT_SIZE for x in input_size]) and \
            not any([fnmatch.fnmatch(model_name, x) for x in EXCLUDE_FILTERS]):
        # pool size only checked if default res <= 448 * 448 to keep resource down
        input_size = tuple([min(x, MAX_FWD_FEAT_SIZE) for x in input_size])
        outputs = model.forward_features(torch.randn((batch_size, *input_size)))
        assert outputs.shape[-1] == pool_size[-1] and outputs.shape[-2] == pool_size[-2]
    assert any([k.startswith(classifier) for k in state_dict.keys()]), f'{classifier} not in model params'
    assert any([k.startswith(first_conv) for k in state_dict.keys()]), f'{first_conv} not in model params'


EXCLUDE_JIT_FILTERS = [
    '*iabn*', 'tresnet*',  # models using inplace abn unlikely to ever be scriptable
    'dla*', 'hrnet*',  # hopefully fix at some point
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=EXCLUDE_FILTERS + EXCLUDE_JIT_FILTERS))
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward_torchscript(model_name, batch_size):
    """Run a single forward pass with each model"""
    with set_scriptable(True):
        model = create_model(model_name, pretrained=False)
    model.eval()
    input_size = (3, 128, 128)  # jit compile is already a bit slow and we've tested normal res already...
    model = torch.jit.script(model)
    outputs = model(torch.randn((batch_size, *input_size)))

    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


EXCLUDE_FEAT_FILTERS = [
    '*pruned*',  # hopefully fix at some point
]
if 'GITHUB_ACTIONS' in os.environ and 'Linux' in platform.system():
    # GitHub Linux runner is slower and hits memory limits sooner than MacOS, exclude bigger models
    EXCLUDE_FEAT_FILTERS += ['*resnext101_32x32d']


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=EXCLUDE_FILTERS + EXCLUDE_FEAT_FILTERS))
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward_features(model_name, batch_size):
    """Run a single forward pass with each model in feature extraction mode"""
    model = create_model(model_name, pretrained=False, features_only=True)
    model.eval()
    expected_channels = model.feature_info.channels()
    assert len(expected_channels) >= 4  # all models here should have at least 4 feature levels by default, some 5 or 6
    input_size = (3, 96, 96)  # jit compile is already a bit slow and we've tested normal res already...
    outputs = model(torch.randn((batch_size, *input_size)))
    assert len(expected_channels) == len(outputs)
    for e, o in zip(expected_channels, outputs):
        assert e == o.shape[1]
        assert o.shape[0] == batch_size
        assert not torch.isnan(o).any()
