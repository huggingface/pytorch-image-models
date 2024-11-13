from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision.ops.misc import FrozenBatchNorm2d

import timm
import pytest
from timm.utils.model import freeze, unfreeze
from timm.utils.model import ActivationStatsHook
from timm.utils.model import extract_spp_stats

from timm.utils.model import _freeze_unfreeze
from timm.utils.model import avg_sq_ch_mean, avg_ch_var, avg_ch_var_residual
from timm.utils.model import reparameterize_model
from timm.utils.model import get_state_dict

def test_freeze_unfreeze():
    model = timm.create_model('resnet18')

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
    model = timm.create_model('resnet18')
    
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
    model = timm.create_model('resnet18')
    
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
    model = timm.create_model('resnet18')
    
    # Test with string input
    _freeze_unfreeze(model, 'layer1', mode='freeze')
    assert model.layer1[0].conv1.weight.requires_grad == False
    
    # Test unfreezing with string input
    _freeze_unfreeze(model, 'layer1', mode='unfreeze')
    assert model.layer1[0].conv1.weight.requires_grad == True

