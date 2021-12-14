""" Conv2d + BN + Act

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import get_norm_act_layer


class ConvNormAct(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
            bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, drop_layer=None):
        super(ConvNormAct, self).__init__()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


ConvBnAct = ConvNormAct


class ConvNormActAa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
            bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, aa_layer=None, drop_layer=None):
        super(ConvNormActAa, self).__init__()
        use_aa = aa_layer is not None

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else nn.Identity()

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.aa(x)
        return x
