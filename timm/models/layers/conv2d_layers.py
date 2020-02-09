from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from itertools import repeat
from functools import partial
import numpy as np
import math


# Tuple helpers ripped from PyTorch
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


# pylint: disable=unused-argument
def conv2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or
                weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


class CondConv2d(nn.Module):
    """ Conditional Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['bias', 'in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic  # if in forward to work with torchscript
        self.padding = _pair(padding_val)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])

        # Literal port (from TF definition)
        # x = torch.split(x, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(x, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out


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


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 drop_block=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(ConvBnAct, self).__init__()
        padding = _get_padding(kernel_size, stride, dilation)  # assuming PyTorch style padding for this block
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


# helper method
def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    assert 'groups' not in kwargs  # only use 'depthwise' bool arg
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else 1
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
    return m
