""" Conv2d + BN + Act

Hacked together by Ross Wightman
"""
from torch import nn as nn

from timm.models.layers import get_padding


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 drop_block=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(ConvBnAct, self).__init__()
        padding = get_padding(kernel_size, stride, dilation)  # assuming PyTorch style padding for this block
        self.conv = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.drop_block = drop_block
        if act_layer is not None:
            self.act = act_layer(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        if self.act is not None:
            x = self.act(x)
        return x
