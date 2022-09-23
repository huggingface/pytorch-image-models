""" Squeeze-and-Excitation Channel Attention

An SE implementation originally based on PyTorch SE-Net impl.
Has since evolved with additional functionality / configuration.

Paper: `Squeeze-and-Excitation Networks` - https://arxiv.org/abs/1709.01507

Also included is Effective Squeeze-Excitation (ESE).
Paper: `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667

Hacked together by / Copyright 2021 Ross Wightman
"""
from torch import nn as nn

from .create_act import create_act_layer
from .helpers import make_divisible


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            bias=True, act_layer=nn.ReLU, norm_layer=None, gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


SqueezeExcite = SEModule  # alias


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """
    def __init__(self, channels, add_maxpool=False, gate_layer='hard_sigmoid', **_):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


EffectiveSqueezeExcite = EffectiveSEModule  # alias


class SqueezeExciteCl(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8,
            bias=True, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super().__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Linear(channels, rd_channels, bias=bias)
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Linear(rd_channels, channels, bias=bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((1, 2), keepdims=True)  # FIXME avg dim [1:n-1], don't assume 2D NHWC
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)