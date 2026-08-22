import pytest
import torch
import timm


@pytest.mark.parametrize('arch', [
    'efficientnet_b3_pruned',
    'efficientnet_b1_pruned',
    'efficientnet_b2_pruned',
    'ecaresnet50d_pruned',
])
@pytest.mark.parametrize('pretrained', [False, True])
def test_pruned_models_in_chans(arch, pretrained):
    # Test creating pruned model with single channel input (in_chans=1)
    model = timm.create_model(arch, pretrained=pretrained, in_chans=1, num_classes=1)
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    assert y.shape == (2, 1)
