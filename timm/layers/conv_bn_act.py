""" Conv2d + BN + Act

Hacked together by / Copyright 2020 Ross Wightman
"""
import functools
from torch import nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import get_norm_act_layer


class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding='',
            dilation=1,
            groups=1,
            bias=False,
            apply_act=True,
            norm_layer=nn.BatchNorm2d,
            norm_kwargs=None,
            act_layer=nn.ReLU,
            act_kwargs=None,
            drop_layer=None,
    ):
        super(ConvNormAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
        if drop_layer:
            norm_kwargs['drop_layer'] = drop_layer
        self.bn = norm_act_layer(
            out_channels,
            apply_act=apply_act,
            act_kwargs=act_kwargs,
            **norm_kwargs,
        )

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


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    if isinstance(aa_layer, functools.partial):
        if issubclass(aa_layer.func, nn.AvgPool2d):
            return aa_layer()
        else:
            return aa_layer(channels)
    elif issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class ConvNormActAa(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding='',
            dilation=1,
            groups=1,
            bias=False,
            apply_act=True,
            norm_layer=nn.BatchNorm2d,
            norm_kwargs=None,
            act_layer=nn.ReLU,
            act_kwargs=None,
            aa_layer=None,
            drop_layer=None,
    ):
        super(ConvNormActAa, self).__init__()
        use_aa = aa_layer is not None and stride == 2
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
        if drop_layer:
            norm_kwargs['drop_layer'] = drop_layer
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, act_kwargs=act_kwargs, **norm_kwargs)
        self.aa = create_aa(aa_layer, out_channels, stride=stride, enable=use_aa)

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
