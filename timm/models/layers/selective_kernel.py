""" Selective Kernel Convolution/Attention

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)

Hacked together by Ross Wightman
"""

import torch
from torch import nn as nn

from .conv_bn_act import ConvBnAct


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernelAttn(nn.Module):
    def __init__(self, channels, num_paths=2, attn_channels=32,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = torch.sum(x, dim=1)
        x = self.pool(x)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        x = torch.softmax(x, dim=1)
        return x


class SelectiveKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, dilation=1, groups=1,
                 attn_reduction=16, min_attn_channels=32, keep_3x3=True, split_input=False,
                 drop_block=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelConv, self).__init__()
        kernel_size = kernel_size or [3, 5]
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)

        conv_kwargs = dict(
            stride=stride, groups=groups, drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer)
        self.paths = nn.ModuleList([
            ConvBnAct(in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs)
            for k, d in zip(kernel_size, dilation)])

        attn_channels = max(int(out_channels / attn_reduction), min_attn_channels)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)
        self.drop_block = drop_block

    def forward(self, x):
        if self.split_input:
            x_split = torch.split(x, self.in_channels // self.num_paths, 1)
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            x_paths = [op(x) for op in self.paths]
        x = torch.stack(x_paths, dim=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = torch.sum(x, dim=1)
        return x
