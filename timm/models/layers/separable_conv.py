from torch import nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import convert_norm_act_type


class SeparableConvBnAct(nn.Module):
    """ Separable Conv w/ trailing Norm and Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 act_layer=nn.ReLU, apply_act=True, drop_block=None):
        super(SeparableConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        norm_act_layer, norm_act_args = convert_norm_act_type(norm_layer, act_layer, norm_kwargs)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block, **norm_act_args)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


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

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x
