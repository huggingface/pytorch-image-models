""" EvoNorm in PyTorch

Based on `Evolving Normalization-Activation Layers` - https://arxiv.org/abs/2004.02967
@inproceedings{NEURIPS2020,
 author = {Liu, Hanxiao and Brock, Andy and Simonyan, Karen and Le, Quoc},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {13539--13550},
 publisher = {Curran Associates, Inc.},
 title = {Evolving Normalization-Activation Layers},
 url = {https://proceedings.neurips.cc/paper/2020/file/9d4c03631b8b0c85ae08bf05eda37d0f-Paper.pdf},
 volume = {33},
 year = {2020}
}

An attempt at getting decent performing EvoNorms running in PyTorch.
While faster than other PyTorch impl, still quite a ways off the built-in BatchNorm
in terms of memory usage and throughput on GPUs.

I'm testing these modules on TPU w/ PyTorch XLA. Promising start but
currently working around some issues with builtin torch/tensor.var/std. Unlike
GPU, similar train speeds for EvoNormS variants and BatchNorm.

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .create_act import create_act_layer
from .trace_utils import _assert


def instance_std(x, eps: float = 1e-5):
    std = x.float().var(dim=(2, 3), unbiased=False, keepdim=True).add(eps).sqrt().to(x.dtype)
    return std.expand(x.shape)


def instance_std_tpu(x, eps: float = 1e-5):
    std = manual_var(x, dim=(2, 3)).add(eps).sqrt()
    return std.expand(x.shape)
# instance_std = instance_std_tpu


def instance_rms(x, eps: float = 1e-5):
    rms = x.float().square().mean(dim=(2, 3), keepdim=True).add(eps).sqrt().to(x.dtype)
    return rms.expand(x.shape)


def manual_var(x, dim: Union[int, Sequence[int]], diff_sqm: bool = False):
    xm = x.mean(dim=dim, keepdim=True)
    if diff_sqm:
        # difference of squared mean and mean squared, faster on TPU can be less stable
        var = ((x * x).mean(dim=dim, keepdim=True) - (xm * xm)).clamp(0)
    else:
        var = ((x - xm) * (x - xm)).mean(dim=dim, keepdim=True)
    return var


def group_std(x, groups: int = 32, eps: float = 1e-5, flatten: bool = False):
    B, C, H, W = x.shape
    x_dtype = x.dtype
    _assert(C % groups == 0, '')
    if flatten:
        x = x.reshape(B, groups, -1)  # FIXME simpler shape causing TPU / XLA issues
        std = x.float().var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
    else:
        x = x.reshape(B, groups, C // groups, H, W)
        std = x.float().var(dim=(2, 3, 4), unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
    return std.expand(x.shape).reshape(B, C, H, W)


def group_std_tpu(x, groups: int = 32, eps: float = 1e-5, diff_sqm: bool = False, flatten: bool = False):
    # This is a workaround for some stability / odd behaviour of .var and .std
    # running on PyTorch XLA w/ TPUs. These manual var impl are producing much better results
    B, C, H, W = x.shape
    _assert(C % groups == 0, '')
    if flatten:
        x = x.reshape(B, groups, -1)  # FIXME simpler shape causing TPU / XLA issues
        var = manual_var(x, dim=-1, diff_sqm=diff_sqm)
    else:
        x = x.reshape(B, groups, C // groups, H, W)
        var = manual_var(x, dim=(2, 3, 4), diff_sqm=diff_sqm)
    return var.add(eps).sqrt().expand(x.shape).reshape(B, C, H, W)
#group_std = group_std_tpu  # FIXME TPU temporary


def group_rms(x, groups: int = 32, eps: float = 1e-5):
    B, C, H, W = x.shape
    _assert(C % groups == 0, '')
    x_dtype = x.dtype
    x = x.reshape(B, groups, C // groups, H, W)
    rms = x.float().square().mean(dim=(2, 3, 4), keepdim=True).add(eps).sqrt_().to(x_dtype)
    return rms.expand(x.shape).reshape(B, C, H, W)


class EvoNorm2dB0(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-3, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.v is not None:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.v is not None:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                # var = manual_var(x, dim=(0, 2, 3)).squeeze()
                n = x.numel() / x.shape[1]
                self.running_var.copy_(
                    self.running_var * (1 - self.momentum) +
                    var.detach() * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            left = var.add(self.eps).sqrt_().to(x_dtype).view(v_shape).expand_as(x)
            v = self.v.to(x_dtype).view(v_shape)
            right = x * v + instance_std(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.to(x_dtype).view(v_shape) + self.bias.to(x_dtype).view(v_shape)


class EvoNorm2dB1(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.apply_act:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                n = x.numel() / x.shape[1]
                self.running_var.copy_(
                    self.running_var * (1 - self.momentum) +
                    var.detach().to(self.running_var.dtype) * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            var = var.to(x_dtype).view(v_shape)
            left = var.add(self.eps).sqrt_()
            right = (x + 1) * instance_rms(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)


class EvoNorm2dB2(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.apply_act:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                n = x.numel() / x.shape[1]
                self.running_var.copy_(
                    self.running_var * (1 - self.momentum) +
                    var.detach().to(self.running_var.dtype) * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            var = var.to(x_dtype).view(v_shape)
            left = var.add(self.eps).sqrt_()
            right = instance_rms(x, self.eps) - x
            x = x / left.max(right)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)


class EvoNorm2dS0(nn.Module):
    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.v is not None:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.v is not None:
            v = self.v.view(v_shape).to(x_dtype)
            x = x * (x * v).sigmoid() / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)


class EvoNorm2dS0a(EvoNorm2dS0):
    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        d = group_std(x, self.groups, self.eps)
        if self.v is not None:
            v = self.v.view(v_shape).to(x_dtype)
            x = x * (x * v).sigmoid()
        x = x / d
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)


class EvoNorm2dS1(nn.Module):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=nn.SiLU, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.pre_act_norm = False
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.apply_act:
            x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)


class EvoNorm2dS1a(EvoNorm2dS1):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=nn.SiLU, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, act_layer=act_layer, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)


class EvoNorm2dS2(nn.Module):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=nn.SiLU, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.apply_act:
            x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)


class EvoNorm2dS2a(EvoNorm2dS2):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=nn.SiLU, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, act_layer=act_layer, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)
