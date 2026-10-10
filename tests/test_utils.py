from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision.ops.misc import FrozenBatchNorm2d

import timm
import torch
import pytest
from timm.utils.model import freeze, unfreeze
from timm.utils.model import ActivationStatsHook
from timm.utils.model import extract_spp_stats

from timm.utils.model import _freeze_unfreeze
from timm.utils.model import avg_sq_ch_mean, avg_ch_var, avg_ch_var_residual
from timm.utils.model import reparameterize_model
from timm.utils.model import get_state_dict
from timm.utils.metrics import AverageMeter
import argparse
from timm.utils.misc import ParseKwargs

def test_average_meter_zero_count():
    # update with n=0 on a fresh meter (count == 0) must not raise ZeroDivisionError
    meter = AverageMeter()
    meter.update(1.0, n=0)
    assert meter.avg == 0


def test_freeze_unfreeze():
    model = timm.create_model('test_resnet', layers=(1, 2, 1, 1))

    # Freeze all
    freeze(model)
    # Check top level module
    assert model.fc.weight.requires_grad == False
    # Check submodule
    assert model.layer1[0].conv1.weight.requires_grad == False
    # Check BN
    assert isinstance(model.layer1[0].bn1, FrozenBatchNorm2d)

    # Unfreeze all
    unfreeze(model)
    # Check top level module
    assert model.fc.weight.requires_grad == True
    # Check submodule
    assert model.layer1[0].conv1.weight.requires_grad == True
    # Check BN
    assert isinstance(model.layer1[0].bn1, BatchNorm2d)

    # Freeze some
    freeze(model, ['layer1', 'layer2.0'])
    # Check frozen
    assert model.layer1[0].conv1.weight.requires_grad == False
    assert isinstance(model.layer1[0].bn1, FrozenBatchNorm2d)
    assert model.layer2[0].conv1.weight.requires_grad == False
    # Check not frozen
    assert model.layer3[0].conv1.weight.requires_grad == True
    assert isinstance(model.layer3[0].bn1, BatchNorm2d)
    assert model.layer2[1].conv1.weight.requires_grad == True

    # Unfreeze some
    unfreeze(model, ['layer1', 'layer2.0'])
    # Check not frozen
    assert model.layer1[0].conv1.weight.requires_grad == True
    assert isinstance(model.layer1[0].bn1, BatchNorm2d)
    assert model.layer2[0].conv1.weight.requires_grad == True

    # Freeze/unfreeze BN
    # From root
    freeze(model, ['layer1.0.bn1'])
    assert isinstance(model.layer1[0].bn1, FrozenBatchNorm2d)
    unfreeze(model, ['layer1.0.bn1'])
    assert isinstance(model.layer1[0].bn1, BatchNorm2d)
    # From direct parent
    freeze(model.layer1[0], ['bn1'])
    assert isinstance(model.layer1[0].bn1, FrozenBatchNorm2d)    
    unfreeze(model.layer1[0], ['bn1'])
    assert isinstance(model.layer1[0].bn1, BatchNorm2d)

def test_activation_stats_hook_validation():
    model = timm.create_model('test_resnet')
    
    def test_hook(model, input, output):
        return output.mean().item()
    
    # Test error case with mismatched lengths
    with pytest.raises(ValueError, match="Please provide `hook_fns` for each `hook_fn_locs`"):
        ActivationStatsHook(
            model,
            hook_fn_locs=['layer1.0.conv1', 'layer1.0.conv2'],
            hook_fns=[test_hook]
        )


def test_extract_spp_stats():
    model = timm.create_model('test_resnet')
    
    def test_hook(model, input, output):
        return output.mean().item()
    
    stats = extract_spp_stats(
        model,
        hook_fn_locs=['layer1.0.conv1'],
        hook_fns=[test_hook],
        input_shape=[2, 3, 32, 32]
    )
    
    assert isinstance(stats, dict)
    assert test_hook.__name__ in stats
    assert isinstance(stats[test_hook.__name__], list)
    assert len(stats[test_hook.__name__]) > 0

def test_freeze_unfreeze_bn_root():
    import torch.nn as nn
    from timm.layers import BatchNormAct2d
    
    # Create batch norm layers
    bn = nn.BatchNorm2d(10)
    bn_act = BatchNormAct2d(10)
    
    # Test with BatchNorm2d as root
    with pytest.raises(AssertionError):
        _freeze_unfreeze(bn, mode="freeze")
    
    # Test with BatchNormAct2d as root
    with pytest.raises(AssertionError):
        _freeze_unfreeze(bn_act, mode="freeze")


def test_activation_stats_functions():
    import torch
    
    # Create sample input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 4, 4)
    
    # Test avg_sq_ch_mean
    result1 = avg_sq_ch_mean(None, None, x)
    assert isinstance(result1, float)
    
    # Test avg_ch_var
    result2 = avg_ch_var(None, None, x)
    assert isinstance(result2, float)
    
    # Test avg_ch_var_residual
    result3 = avg_ch_var_residual(None, None, x)
    assert isinstance(result3, float)


def test_reparameterize_model():
    import torch.nn as nn
    
    class FusableModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)
        
        def fuse(self):
            return nn.Identity()
    
    class ModelWithFusable(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusable = FusableModule()
            self.normal = nn.Linear(10, 10)
    
    model = ModelWithFusable()
    
    # Test with inplace=False (should create a copy)
    new_model = reparameterize_model(model, inplace=False)
    assert isinstance(new_model.fusable, nn.Identity)
    assert isinstance(model.fusable, FusableModule)  # Original unchanged
    
    # Test with inplace=True
    reparameterize_model(model, inplace=True)
    assert isinstance(model.fusable, nn.Identity)


def test_get_state_dict_custom_unwrap():
    import torch.nn as nn
    
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
    
    model = CustomModel()
    
    def custom_unwrap(m):
        return m
    
    state_dict = get_state_dict(model, unwrap_fn=custom_unwrap)
    assert 'linear.weight' in state_dict
    assert 'linear.bias' in state_dict


def test_freeze_unfreeze_string_input():
    model = timm.create_model('test_resnet')
    
    # Test with string input
    _freeze_unfreeze(model, 'layer1', mode='freeze')
    assert model.layer1[0].conv1.weight.requires_grad == False
    
    # Test unfreezing with string input
    _freeze_unfreeze(model, 'layer1', mode='unfreeze')
    assert model.layer1[0].conv1.weight.requires_grad == True


def _parse_model_kwargs(tokens):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)
    args = parser.parse_args(['--model-kwargs'] + tokens)
    return args.model_kwargs


def test_parse_kwargs_literal_values():
    assert _parse_model_kwargs(['depth=12', 'drop_rate=0.1', 'pretrained=True']) == {
        'depth': 12,
        'drop_rate': 0.1,
        'pretrained': True,
    }


def test_parse_kwargs_value_with_equals():
    # A value that itself contains '=' must split only on the first '=' and be
    # kept verbatim. Before, split('=') raised "too many values to unpack", and
    # the literal_eval fallback only caught ValueError so 'size=large' (invalid
    # syntax) still raised SyntaxError.
    assert _parse_model_kwargs(['note=size=large', 'url=http://h/p?a=1&b=2']) == {
        'note': 'size=large',
        'url': 'http://h/p?a=1&b=2',
    }


def test_parse_kwargs_non_literal_falls_back_to_string():
    assert _parse_model_kwargs(['act=nn.GELU', 'cfg={"a": 1}']) == {
        'act': 'nn.GELU',
        'cfg': {'a': 1},
    }


def _ema_drift_test(model, ema, steps=2000):
    """Drift a model's weights, update ema, return (ema drift, float64 reference drift) as mean per-element."""
    start = model.weight.detach().double().clone()
    ref = start.clone()  # float64 reference EMA over the model's actual (possibly low precision) weights
    for step in range(steps):
        with torch.no_grad():
            model.weight.add_(1e-3 * (1 + torch.arange(model.weight.shape[0]).view(-1, 1) % 2))
        ema.update(model, step=step)
        ref = ref.lerp(model.weight.detach().double(), 1. - ema.get_decay(step))
    drift_ema = (ema.module.weight.double() - start).mean().item()
    drift_ref = (ref - start).mean().item()
    return drift_ema, drift_ref


@pytest.mark.parametrize('foreach', [True, False])
@pytest.mark.parametrize('exclude_buffers', [True, False])
def test_model_ema_v3_bf16_model_fp32_ema(foreach, exclude_buffers):
    """A float32 EMA of a bfloat16 model tracks a float64 reference (mixed dtype update paths)."""
    from timm.utils.model_ema import ModelEmaV3
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16).to(torch.bfloat16)
    ema = ModelEmaV3(model, decay=0.9998, foreach=foreach, exclude_buffers=exclude_buffers, dtype=torch.float32)
    assert ema.module.weight.dtype == torch.float32
    drift_ema, drift_ref = _ema_drift_test(model, ema)
    assert abs(drift_ema - drift_ref) < 0.01 * drift_ref


@pytest.mark.parametrize('foreach', [True, False])
@pytest.mark.parametrize('exclude_buffers', [True, False])
def test_model_ema_v3_bf16_ema_stochastic_rounding(foreach, exclude_buffers):
    """A bfloat16 EMA w/ stochastic rounding tracks the reference, without it the update rounds away (default
    decay ~0.9998 is far below bfloat16 resolution) and the EMA barely moves."""
    from timm.utils.model_ema import ModelEmaV3
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16).to(torch.bfloat16)
    ema = ModelEmaV3(model, decay=0.9998, foreach=foreach, exclude_buffers=exclude_buffers)
    assert ema.module.weight.dtype == torch.bfloat16
    drift_ema, drift_ref = _ema_drift_test(model, ema)
    assert abs(drift_ema - drift_ref) < 0.05 * drift_ref

    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16).to(torch.bfloat16)
    ema = ModelEmaV3(model, decay=0.9998, foreach=foreach, exclude_buffers=exclude_buffers, stochastic_rounding=False)
    drift_ema, drift_ref = _ema_drift_test(model, ema)
    assert drift_ema < 0.2 * drift_ref


def test_model_ema_v3_fp32_unchanged():
    """float32 model & EMA, foreach and per-tensor paths match each other and the float64 reference."""
    from timm.utils.model_ema import ModelEmaV3
    results = []
    for foreach in (True, False):
        torch.manual_seed(0)
        model = torch.nn.Linear(16, 16)
        ema = ModelEmaV3(model, decay=0.9998, foreach=foreach)
        assert ema.module.weight.dtype == torch.float32
        drift_ema, drift_ref = _ema_drift_test(model, ema, steps=200)
        assert abs(drift_ema - drift_ref) < 1e-5 * drift_ref
        results.append(ema.module.weight.clone())
    torch.testing.assert_close(results[0], results[1])


def test_model_ema_v3_bf16_stochastic_rounding_unbiased():
    """Repeated stochastic rounding of the same sub-resolution update averages to the exact value."""
    from timm.utils.model_ema import _lerp_stochastic_round_bf16_
    torch.manual_seed(0)
    ema = torch.ones(100000, dtype=torch.bfloat16)
    model = torch.full_like(ema, 2.)
    weight = 0.001  # exact result 1.001, below bfloat16 resolution (0.0078 at 1.0)
    _lerp_stochastic_round_bf16_([ema], [model], weight)
    assert set(ema.unique().tolist()) == {1.0, 1.0078125}
    assert abs(ema.double().mean().item() - 1.001) < 1e-4


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')),
])
@pytest.mark.parametrize('foreach', [True, False])
@pytest.mark.parametrize('exclude_buffers', [True, False])
def test_model_ema_v3_stochastic_rounding_rank_rng(device, foreach, exclude_buffers):
    """Rank-local randomness does not affect rounding, and EMA does not consume the global RNG."""
    from timm.utils.model_ema import ModelEmaV3
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.BatchNorm1d(64)).to(
        device=device, dtype=torch.bfloat16,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.)
    kwargs = dict(decay=0.9998, foreach=foreach, exclude_buffers=exclude_buffers)
    emas = [ModelEmaV3(model, stochastic_rounding_seed=17, **kwargs) for _ in range(2)]
    other_seed = ModelEmaV3(model, stochastic_rounding_seed=18, **kwargs)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(2.)
        model[1].running_mean.fill_(2.)
        model[1].num_batches_tracked.fill_(10)

    for step in range(10, 14):
        for rank, ema in enumerate(emas):
            torch.manual_seed(100 * step + rank)
            torch.rand(13 * (rank + 1), device=device)
            cpu_rng = torch.get_rng_state()
            if device == 'cuda':
                cuda_rng = torch.cuda.get_rng_state()
            ema.update(model, step=step)
            assert torch.equal(torch.get_rng_state(), cpu_rng)
            if device == 'cuda':
                assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        other_seed.update(model, step=step)
        for value0, value1 in zip(emas[0].module.state_dict().values(), emas[1].module.state_dict().values()):
            assert torch.equal(value0, value1)
    assert not torch.equal(emas[0].module[0].weight, other_seed.module[0].weight)


@pytest.mark.parametrize('foreach', [True, False])
def test_model_ema_v3_stochastic_rounding_advances(foreach):
    """Different tensors and successive updates receive different rounding randomness."""
    from timm.utils.model_ema import ModelEmaV3
    model = torch.nn.Sequential(*(torch.nn.Linear(64, 64, bias=False) for _ in range(2))).to(torch.bfloat16)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(2.)
    ema = ModelEmaV3(model, decay=0.9998, foreach=foreach)
    results = []
    for step in (10, 11):
        with torch.no_grad():
            for param in ema.module.parameters():
                param.fill_(1.)
        ema.update(model, step=step)
        assert not torch.equal(ema.module[0].weight, ema.module[1].weight)
        results.append(ema.module[0].weight.detach().clone())
    assert not torch.equal(results[0], results[1])
