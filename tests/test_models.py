import pytest
import torch
import platform
import os
import fnmatch

import timm
from timm import list_models, create_model, set_scriptable

if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)

# transformer models don't support many of the spatial / feature based model functionalities
NON_STD_FILTERS = ['vit_*']

# exclude models that cause specific test failures
if 'GITHUB_ACTIONS' in os.environ:  # and 'Linux' in platform.system():
    # GitHub Linux runner is slower and hits memory limits sooner than MacOS, exclude bigger models
    EXCLUDE_FILTERS = [
        '*efficientnet_l2*', '*resnext101_32x48d', '*in21k', '*152x4_bitm',
        '*nfnet_f4*', '*nfnet_f5*', '*nfnet_f6*', '*nfnet_f7*'] + NON_STD_FILTERS
else:
    EXCLUDE_FILTERS = NON_STD_FILTERS

MAX_FWD_SIZE = 384
MAX_BWD_SIZE = 128
MAX_FWD_FEAT_SIZE = 448


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=EXCLUDE_FILTERS[:-1]))
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
@pytest.mark.parametrize('model_name', list_models(exclude_filters=NON_STD_FILTERS))
@pytest.mark.parametrize('batch_size', [1])
def test_model_default_cfgs(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False)
    model.eval()
    state_dict = model.state_dict()
    cfg = model.default_cfg

    classifier = cfg['classifier']
    pool_size = cfg['pool_size']
    input_size = model.default_cfg['input_size']

    if all([x <= MAX_FWD_FEAT_SIZE for x in input_size]) and \
            not any([fnmatch.fnmatch(model_name, x) for x in EXCLUDE_FILTERS]):
        # output sizes only checked if default res <= 448 * 448 to keep resource down
        input_size = tuple([min(x, MAX_FWD_FEAT_SIZE) for x in input_size])
        input_tensor = torch.randn((batch_size, *input_size))

        # test forward_features (always unpooled)
        outputs = model.forward_features(input_tensor)
        assert outputs.shape[-1] == pool_size[-1] and outputs.shape[-2] == pool_size[-2]

        # test forward after deleting the classifier, output should be poooled, size(-1) == model.num_features
        model.reset_classifier(0)
        outputs = model.forward(input_tensor)
        assert len(outputs.shape) == 2
        assert outputs.shape[-1] == model.num_features

        # test model forward without pooling and classifier
        model.reset_classifier(0, '')  # reset classifier and set global pooling to pass-through
        outputs = model.forward(input_tensor)
        assert len(outputs.shape) == 4
        if not isinstance(model, timm.models.MobileNetV3):
            # FIXME mobilenetv3 forward_features vs removed pooling differ
            assert outputs.shape[-1] == pool_size[-1] and outputs.shape[-2] == pool_size[-2]

    # check classifier name matches default_cfg
    assert classifier + ".weight" in state_dict.keys(), f'{classifier} not in model params'

    # check first conv(s) names match default_cfg
    first_conv = cfg['first_conv']
    if isinstance(first_conv, str):
        first_conv = (first_conv,)
    assert isinstance(first_conv, (tuple, list))
    for fc in first_conv:
        assert fc + ".weight" in state_dict.keys(), f'{fc} not in model params'


if 'GITHUB_ACTIONS' not in os.environ:
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize('model_name', list_models(pretrained=True))
    @pytest.mark.parametrize('batch_size', [1])
    def test_model_load_pretrained(model_name, batch_size):
        """Create that pretrained weights load, verify support for in_chans != 3 while doing so."""
        in_chans = 3 if 'pruned' in model_name else 1  # pruning not currently supported with in_chans change
        create_model(model_name, pretrained=True, in_chans=in_chans)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize('model_name', list_models(pretrained=True, exclude_filters=NON_STD_FILTERS))
    @pytest.mark.parametrize('batch_size', [1])
    def test_model_features_pretrained(model_name, batch_size):
        """Create that pretrained weights load when features_only==True."""
        create_model(model_name, pretrained=True, features_only=True)

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
if 'GITHUB_ACTIONS' in os.environ:  # and 'Linux' in platform.system():
    # GitHub Linux runner is slower and hits memory limits sooner than MacOS, exclude bigger models
    EXCLUDE_FEAT_FILTERS += ['*resnext101_32x32d', '*resnext101_32x16d']


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
