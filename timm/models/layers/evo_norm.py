"""EvoNormB0 (Batched) and EvoNormS0 (Sample) in PyTorch

An attempt at getting decent performing EvoNorms running in PyTorch.
While currently faster than other impl, still quite a ways off the built-in BN
in terms of memory usage and throughput (roughly 5x mem, 1/2 - 1/3x speed).

Still very much a WIP, fiddling with buffer usage, in-place/jit optimizations, and layouts.

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
import torch.nn as nn

from .trace_utils import _assert


class EvoNormBatch2d(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, drop_block=None):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.v = nn.Parameter(torch.ones(num_features), requires_grad=True) if apply_act else None
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_type = x.dtype
        if self.v is not None:
            running_var = self.running_var.view(1, -1, 1, 1)
            if self.training:
                var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
                n = x.numel() / x.shape[1]
                running_var = var.detach() * self.momentum * (n / (n - 1)) + running_var * (1 - self.momentum)
                self.running_var.copy_(running_var.view(self.running_var.shape))
            else:
                var = running_var
            v = self.v.to(dtype=x_type).reshape(1, -1, 1, 1)
            d = x * v + (x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt().to(dtype=x_type)
            d = d.max((var + self.eps).sqrt().to(dtype=x_type))
            x = x / d
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class EvoNormSample2d(nn.Module):
    def __init__(self, num_features, apply_act=True, groups=32, eps=1e-5, drop_block=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.v = nn.Parameter(torch.ones(num_features), requires_grad=True) if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        B, C, H, W = x.shape
        _assert(C % self.groups == 0, '')
        if self.v is not None:
            n = x * (x * self.v.view(1, -1, 1, 1)).sigmoid()
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            x = x.reshape(B, C, H, W)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
