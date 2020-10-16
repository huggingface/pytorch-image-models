""" Selective Kernel Convolution/Attention

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)

Hacked together by / Copyright 2020 Ross Wightman
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
        """ Selective Kernel Attention Module

        Selective Kernel attention mechanism factored out into its own module.

        """
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = x.sum(1).mean((2, 3), keepdim=True)
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
                 drop_block=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None):
        """ Selective Kernel Convolution Module

        As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.

        Largest change is the input split, which divides the input channels across each convolution path, this can
        be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
        the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
        a noteworthy increase in performance over similar param count models without this attention layer. -Ross W

        Args:
            in_channels (int):  module input (feature) channel count
            out_channels (int):  module output (feature) channel count
            kernel_size (int, list): kernel size for each convolution branch
            stride (int): stride for convolutions
            dilation (int): dilation for module as a whole, impacts dilation of each branch
            groups (int): number of groups for each branch
            attn_reduction (int, float): reduction factor for attention features
            min_attn_channels (int): minimum attention feature channels
            keep_3x3 (bool): keep all branch convolution kernels as 3x3, changing larger kernels for dilations
            split_input (bool): split input channels evenly across each convolution branch, keeps param count lower,
                can be viewed as grouping by path, output expands to module out_channels count
            drop_block (nn.Module): drop block module
            act_layer (nn.Module): activation layer to use
            norm_layer (nn.Module): batchnorm/norm layer to use
        """
        super(SelectiveKernelConv, self).__init__()
        kernel_size = kernel_size or [3, 5]  # default to one 3x3 and one 5x5 branch. 5x5 -> 3x3 + dilation
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
            stride=stride, groups=groups, drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer,
            aa_layer=aa_layer)
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
