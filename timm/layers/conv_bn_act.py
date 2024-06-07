""" Conv2d + BN + Act

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import Any, Dict, Optional, Type

from torch import nn as nn

from .typing import LayerType, PadType
from .blur_pool import create_aa
from .create_conv2d import create_conv2d
from .create_norm_act import get_norm_act_layer


class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: PadType = '',
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            apply_norm: bool = True,
            apply_act: bool = True,
            norm_layer: LayerType = nn.BatchNorm2d,
            act_layer: LayerType = nn.ReLU,
            drop_layer: Optional[Type[nn.Module]] = None,
            conv_kwargs: Optional[Dict[str, Any]] = None,
            norm_kwargs: Optional[Dict[str, Any]] = None,
            act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ConvNormAct, self).__init__()
        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **conv_kwargs,
        )

        if apply_norm:
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
        else:
            self.bn = nn.Sequential()
            if drop_layer:
                norm_kwargs['drop_layer'] = drop_layer
                self.bn.add_module('drop', drop_layer())

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
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: PadType = '',
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            apply_norm: bool = True,
            apply_act: bool = True,
            norm_layer: LayerType = nn.BatchNorm2d,
            act_layer: LayerType = nn.ReLU,
            aa_layer: Optional[LayerType] = None,
            drop_layer: Optional[Type[nn.Module]] = None,
            conv_kwargs: Optional[Dict[str, Any]] = None,
            norm_kwargs: Optional[Dict[str, Any]] = None,
            act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ConvNormActAa, self).__init__()
        use_aa = aa_layer is not None and stride == 2
        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size,
            stride=1 if use_aa else stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **conv_kwargs,
        )

        if apply_norm:
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
        else:
            self.bn = nn.Sequential()
            if drop_layer:
                norm_kwargs['drop_layer'] = drop_layer
                self.bn.add_module('drop', drop_layer())

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
