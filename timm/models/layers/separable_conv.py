""" Depthwise Separable Conv Modules

Basic DWS convs. Other variations of DWS exist with batch norm or activations between the
DW and PW convs such as the Depthwise modules in MobileNetV2 / EfficientNet and Xception.

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import get_norm_act_layer


class SeparableConvNormAct(nn.Module):
    """ Separable Conv w/ trailing Norm and Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                 apply_act=True, drop_layer=None):
        super(SeparableConvNormAct, self).__init__()

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.bn(x)
        return x


SeparableConvBnAct = SeparableConvNormAct


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1):
        super(SeparableConv2d, self).__init__()

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x
