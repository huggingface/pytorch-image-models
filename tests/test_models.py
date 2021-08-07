import pytest
import torch
import platform
import os
import fnmatch

import timm
from timm import list_models, create_model, set_scriptable, has_model_default_key, is_model_default_key, \
    get_model_default_value

if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)

# transformer models don't support many of the spatial / feature based model functionalities
NON_STD_FILTERS = [
    'vit_*', 'tnt_*', 'pit_*', 'swin_*', 'coat_*', 'cait_*', '*mixer_*', 'gmlp_*', 'resmlp_*', 'twins_*',
    'convit_*', 'levit*', 'visformer*', 'deit*', 'jx_nest_*', 'nest_*', 'xcit_*']
NUM_NON_STD = len(NON_STD_FILTERS)

# exclude models that cause specific test failures
if 'GITHUB_ACTIONS' in os.environ:  # and 'Linux' in platform.system():
    # GitHub Linux runner is slower and hits memory limits sooner than MacOS, exclude bigger models
    EXCLUDE_FILTERS = [
        '*efficientnet_l2*', '*resnext101_32x48d', '*in21k', '*152x4_bitm', '*101x3_bitm', '*50x3_bitm',
        '*nfnet_f3*', '*nfnet_f4*', '*nfnet_f5*', '*nfnet_f6*', '*nfnet_f7*', '*efficientnetv2_xl*',
        '*resnetrs350*', '*resnetrs420*', 'xcit_large_24_p8*']
else:
    EXCLUDE_FILTERS = []

TARGET_FWD_SIZE = MAX_FWD_SIZE = 384
TARGET_BWD_SIZE = 128
MAX_BWD_SIZE = 320
MAX_FWD_OUT_SIZE = 448
TARGET_JIT_SIZE = 128
MAX_JIT_SIZE = 320
TARGET_FFEAT_SIZE = 96
MAX_FFEAT_SIZE = 256


def _get_input_size(model=None, model_name='', target=None):
    if model is None:
        assert model_name, "One of model or model_name must be provided"
        input_size = get_model_default_value(model_name, 'input_size')
        fixed_input_size = get_model_default_value(model_name, 'fixed_input_size')
        min_input_size = get_model_default_value(model_name, 'min_input_size')
    else:
        default_cfg = model.default_cfg
        input_size = default_cfg['input_size']
        fixed_input_size = default_cfg.get('fixed_input_size', None)
        min_input_size = default_cfg.get('min_input_size', None)
    assert input_size is not None

    if fixed_input_size:
        return input_size

    if min_input_size:
        if target and max(input_size) > target:
            input_size = min_input_size
    else:
        if target and max(input_size) > target:
            input_size = tuple([min(x, target) for x in input_size])
    return input_size


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=EXCLUDE_FILTERS))
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False)
    model.eval()

    input_size = _get_input_size(model=model, target=TARGET_FWD_SIZE)
    if max(input_size) > MAX_FWD_SIZE:
        pytest.skip("Fixed input size model > limit.")
    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)

    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=EXCLUDE_FILTERS, name_matches_cfg=True))
@pytest.mark.parametrize('batch_size', [2])
def test_model_backward(model_name, batch_size):
    """Run a single forward pass with each model"""
    input_size = _get_input_size(model_name=model_name, target=TARGET_BWD_SIZE)
    if max(input_size) > MAX_BWD_SIZE:
        pytest.skip("Fixed input size model > limit.")

    model = create_model(model_name, pretrained=False, num_classes=42)
    num_params = sum([x.numel() for x in model.parameters()])
    model.train()

    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = torch.cat(outputs)
    outputs.mean().backward()
    for n, x in model.named_parameters():
        assert x.grad is not None, f'No gradient for {n}'
    num_grad = sum([x.grad.numel() for x in model.parameters() if x.grad is not None])

    assert outputs.shape[-1] == 42
    assert num_params == num_grad, 'Some parameters are missing gradients'
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


@pytest.mark.timeout(300)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=NON_STD_FILTERS))
@pytest.mark.parametrize('batch_size', [1])
def test_model_default_cfgs(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False)
    model.eval()
    state_dict = model.state_dict()
    cfg = model.default_cfg

    pool_size = cfg['pool_size']
    input_size = model.default_cfg['input_size']

    if all([x <= MAX_FWD_OUT_SIZE for x in input_size]) and \
            not any([fnmatch.fnmatch(model_name, x) for x in EXCLUDE_FILTERS]):
        # output sizes only checked if default res <= 448 * 448 to keep resource down
        input_size = tuple([min(x, MAX_FWD_OUT_SIZE) for x in input_size])
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
        if not isinstance(model, timm.models.MobileNetV3) and not isinstance(model, timm.models.GhostNet):
            # FIXME mobilenetv3/ghostnet forward_features vs removed pooling differ
            assert outputs.shape[-1] == pool_size[-1] and outputs.shape[-2] == pool_size[-2]

        if 'pruned' not in model_name:  # FIXME better pruned model handling
            # test classifier + global pool deletion via __init__
            model = create_model(model_name, pretrained=False, num_classes=0, global_pool='').eval()
            outputs = model.forward(input_tensor)
            assert len(outputs.shape) == 4
            if not isinstance(model, timm.models.MobileNetV3) and not isinstance(model, timm.models.GhostNet):
                # FIXME mobilenetv3/ghostnet forward_features vs removed pooling differ
                assert outputs.shape[-1] == pool_size[-1] and outputs.shape[-2] == pool_size[-2]

    # check classifier name matches default_cfg
    classifier = cfg['classifier']
    if not isinstance(classifier, (tuple, list)):
        classifier = classifier,
    for c in classifier:
        assert c + ".weight" in state_dict.keys(), f'{c} not in model params'

    # check first conv(s) names match default_cfg
    first_conv = cfg['first_conv']
    if isinstance(first_conv, str):
        first_conv = (first_conv,)
    assert isinstance(first_conv, (tuple, list))
    for fc in first_conv:
        assert fc + ".weight" in state_dict.keys(), f'{fc} not in model params'


@pytest.mark.timeout(300)
@pytest.mark.parametrize('model_name', list_models(filter=NON_STD_FILTERS))
@pytest.mark.parametrize('batch_size', [1])
def test_model_default_cfgs_non_std(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False)
    model.eval()
    state_dict = model.state_dict()
    cfg = model.default_cfg

    input_size = _get_input_size(model=model)
    if max(input_size) > 320:  # FIXME const
        pytest.skip("Fixed input size model > limit.")

    input_tensor = torch.randn((batch_size, *input_size))

    # test forward_features (always unpooled)
    outputs = model.forward_features(input_tensor)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    assert outputs.shape[1] == model.num_features

    # test forward after deleting the classifier, output should be poooled, size(-1) == model.num_features
    model.reset_classifier(0)
    outputs = model.forward(input_tensor)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    assert len(outputs.shape) == 2
    assert outputs.shape[1] == model.num_features

    model = create_model(model_name, pretrained=False, num_classes=0).eval()
    outputs = model.forward(input_tensor)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    assert len(outputs.shape) == 2
    assert outputs.shape[1] == model.num_features

    # check classifier name matches default_cfg
    classifier = cfg['classifier']
    if not isinstance(classifier, (tuple, list)):
        classifier = classifier,
    for c in classifier:
        assert c + ".weight" in state_dict.keys(), f'{c} not in model params'

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
        create_model(model_name, pretrained=True, in_chans=in_chans, num_classes=5)
        create_model(model_name, pretrained=True, in_chans=in_chans, num_classes=0)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize('model_name', list_models(pretrained=True, exclude_filters=NON_STD_FILTERS))
    @pytest.mark.parametrize('batch_size', [1])
    def test_model_features_pretrained(model_name, batch_size):
        """Create that pretrained weights load when features_only==True."""
        create_model(model_name, pretrained=True, features_only=True)

EXCLUDE_JIT_FILTERS = [
    '*iabn*', 'tresnet*',  # models using inplace abn unlikely to ever be scriptable
    'dla*', 'hrnet*', 'ghostnet*',  # hopefully fix at some point
    'vit_large_*', 'vit_huge_*',
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    'model_name', list_models(exclude_filters=EXCLUDE_FILTERS + EXCLUDE_JIT_FILTERS, name_matches_cfg=True))
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward_torchscript(model_name, batch_size):
    """Run a single forward pass with each model"""
    input_size = _get_input_size(model_name=model_name, target=TARGET_JIT_SIZE)
    if max(input_size) > MAX_JIT_SIZE:
        pytest.skip("Fixed input size model > limit.")

    with set_scriptable(True):
        model = create_model(model_name, pretrained=False)
    model.eval()

    model = torch.jit.script(model)
    outputs = model(torch.randn((batch_size, *input_size)))

    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


EXCLUDE_FEAT_FILTERS = [
    '*pruned*',  # hopefully fix at some point
] + NON_STD_FILTERS
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

    input_size = _get_input_size(model=model, target=TARGET_FFEAT_SIZE)
    if max(input_size) > MAX_FFEAT_SIZE:
        pytest.skip("Fixed input size model > limit.")

    outputs = model(torch.randn((batch_size, *input_size)))
    assert len(expected_channels) == len(outputs)
    for e, o in zip(expected_channels, outputs):
        assert e == o.shape[1]
        assert o.shape[0] == batch_size
        assert not torch.isnan(o).any()
