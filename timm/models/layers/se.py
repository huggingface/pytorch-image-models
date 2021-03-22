from torch import nn as nn
import torch.nn.functional as F

from .create_act import create_act_layer
from .helpers import make_divisible


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * min_channels can be specified to keep reduced channel count at a minimum (default: 8)
        * divisor can be specified to keep channels rounded to specified values (default: 1)
        * reduction channels can be specified directly by arg (if reduction_channels is set)
        * reduction channels can be specified by float ratio (if reduction_ratio is set)
    """
    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, gate_layer='sigmoid',
                 reduction_ratio=None, reduction_channels=None, min_channels=8, divisor=1):
        super(SEModule, self).__init__()
        if reduction_channels is not None:
            reduction_channels = reduction_channels  # direct specification highest priority, no rounding/min done
        elif reduction_ratio is not None:
            reduction_channels = make_divisible(channels * reduction_ratio, divisor, min_channels)
        else:
            reduction_channels = make_divisible(channels // reduction, divisor, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """
    def __init__(self, channels, gate_layer='hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer, inplace=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)
