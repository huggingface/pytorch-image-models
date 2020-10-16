""" CBAM (sort-of) Attention

Experimental impl of CBAM: Convolutional Block Attention Module: https://arxiv.org/abs/1807.06521

WARNING: Results with these attention layers have been mixed. They can significantly reduce performance on
some tasks, especially fine-grained it seems. I may end up removing this impl.

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
import torch.nn.functional as F
from .conv_bn_act import ConvBnAct


class ChannelAttn(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """
    def __init__(self, channels, reduction=16, act_layer=nn.ReLU):
        super(ChannelAttn, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x):
        x_avg = x.mean((2, 3), keepdim=True)
        x_max = F.adaptive_max_pool2d(x, 1)
        x_avg = self.fc2(self.act(self.fc1(x_avg)))
        x_max = self.fc2(self.act(self.fc1(x_max)))
        x_attn = x_avg + x_max
        return x * x_attn.sigmoid()


class LightChannelAttn(ChannelAttn):
    """An experimental 'lightweight' that sums avg + max pool first
    """
    def __init__(self, channels, reduction=16):
        super(LightChannelAttn, self).__init__(channels, reduction)

    def forward(self, x):
        x_pool = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * F.adaptive_max_pool2d(x, 1)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * x_attn.sigmoid()


class SpatialAttn(nn.Module):
    """ Original CBAM spatial attention module
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBnAct(2, 1, kernel_size, act_layer=None)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_attn = torch.cat([x_avg, x_max], dim=1)
        x_attn = self.conv(x_attn)
        return x * x_attn.sigmoid()


class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """
    def __init__(self, kernel_size=7):
        super(LightSpatialAttn, self).__init__()
        self.conv = ConvBnAct(1, 1, kernel_size, act_layer=None)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_attn = 0.5 * x_avg + 0.5 * x_max
        x_attn = self.conv(x_attn)
        return x * x_attn.sigmoid()


class CbamModule(nn.Module):
    def __init__(self, channels, spatial_kernel_size=7):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(channels)
        self.spatial = SpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class LightCbamModule(nn.Module):
    def __init__(self, channels, spatial_kernel_size=7):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(channels)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

