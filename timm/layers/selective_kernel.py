""" Selective Kernel Convolution/Attention

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn as nn

from .conv_bn_act import ConvNormAct
from .helpers import make_divisible
from .trace_utils import _assert


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernelAttn(nn.Module):
    def __init__(
            self,
            channels: int,
            num_paths: int = 2,
            attn_channels: int = 32,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            device=None,
            dtype=None,
    ):
        """ Selective Kernel Attention Module

        Selective Kernel attention mechanism factored out into its own module.

        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False, **dd)
        self.bn = norm_layer(attn_channels, **dd)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False, **dd)

    def forward(self, x):
        _assert(x.shape[1] == self.num_paths, '')
        x = x.sum(1).mean((2, 3), keepdim=True)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        x = torch.softmax(x, dim=1)
        return x


class SelectiveKernel(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_size: Optional[Union[int, List[int]]] = None,
            stride: int = 1,
            dilation: int = 1,
            groups: int = 1,
            rd_ratio: float = 1./16,
            rd_channels: Optional[int] = None,
            rd_divisor: int = 8,
            keep_3x3: bool = True,
            split_input: bool = True,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module]= nn.BatchNorm2d,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None,
    ):
        """ Selective Kernel Convolution Module

        As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.

        Largest change is the input split, which divides the input channels across each convolution path, this can
        be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
        the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
        a noteworthy increase in performance over similar param count models without this attention layer. -Ross W

        Args:
            in_channels:  module input (feature) channel count
            out_channels:  module output (feature) channel count
            kernel_size: kernel size for each convolution branch
            stride: stride for convolutions
            dilation: dilation for module as a whole, impacts dilation of each branch
            groups: number of groups for each branch
            rd_ratio: reduction factor for attention features
            keep_3x3: keep all branch convolution kernels as 3x3, changing larger kernels for dilations
            split_input: split input channels evenly across each convolution branch, keeps param count lower,
                can be viewed as grouping by path, output expands to module out_channels count
            act_layer: activation layer to use
            norm_layer: batchnorm/norm layer to use
            aa_layer: anti-aliasing module
            drop_layer: spatial drop module in convs (drop block, etc)
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_channels = out_channels or in_channels
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
            stride=stride, groups=groups, act_layer=act_layer, norm_layer=norm_layer,
            aa_layer=aa_layer, drop_layer=drop_layer, **dd)
        self.paths = nn.ModuleList([
            ConvNormAct(in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs)
            for k, d in zip(kernel_size, dilation)])

        attn_channels = rd_channels or make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels, **dd)

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
