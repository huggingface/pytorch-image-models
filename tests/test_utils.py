from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision.ops.misc import FrozenBatchNorm2d

import timm
from timm.utils.model import freeze, unfreeze


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