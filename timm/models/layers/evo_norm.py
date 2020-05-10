"""EvoNormB0 (Batched) and EvoNormS0 (Sample) in PyTorch

An attempt at getting decent performing EvoNorms running in PyTorch.
While currently faster than other impl, still quite a ways off the built-in BN
in terms of memory usage and throughput.

Still very much a WIP, fiddling with buffer usage, in-place optimizations, and layouts.

Hacked together by Ross Wightman
"""

import torch
import torch.nn as nn


@torch.jit.script
def evo_batch_jit(
        x: torch.Tensor, v: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, running_var: torch.Tensor,
        momentum: float, training: bool, nonlin: bool, eps: float):
    x_type = x.dtype
    running_var = running_var.detach()  # FIXME why is this needed, it's a buffer?
    if training:
        var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)  # FIXME biased, unbiased?
        running_var.copy_(momentum * var + (1 - momentum) * running_var)
    else:
        var = running_var.clone()

    if nonlin:
        # FIXME biased, unbiased?
        d = (x * v.to(x_type)) + x.var(dim=(2, 3), unbiased=False, keepdim=True).add_(eps).sqrt_().to(dtype=x_type)
        d = d.max(var.add(eps).sqrt_().to(dtype=x_type))
        x = x / d
        return x.mul_(weight).add_(bias)
    else:
        return x.mul(weight).add_(bias)


class EvoNormBatch2d(nn.Module):
    def __init__(self, num_features, momentum=0.1, nonlin=True, eps=1e-5, jit=True):
        super(EvoNormBatch2d, self).__init__()
        self.momentum = momentum
        self.nonlin = nonlin
        self.eps = eps
        self.jit = jit
        param_shape = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if nonlin:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.nonlin:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'

        if self.jit:
            return evo_batch_jit(
                x, self.v, self.weight, self.bias, self.running_var, self.momentum,
                self.training, self.nonlin, self.eps)
        else:
            x_type = x.dtype
            if self.training:
                var = x.var(dim=(0, 2, 3), keepdim=True)
                self.running_var.copy_(self.momentum * var + (1 - self.momentum) * self.running_var)
            else:
                var = self.running_var.clone()

            if self.nonlin:
                v = self.v.to(dtype=x_type)
                d = (x * v) + x.var(dim=(2, 3), keepdim=True).add_(self.eps).sqrt_().to(dtype=x_type)
                d = d.max(var.add(self.eps).sqrt_().to(dtype=x_type))
                x = x / d
                return x.mul_(self.weight).add_(self.bias)
            else:
                return x.mul(self.weight).add_(self.bias)


@torch.jit.script
def evo_sample_jit(
        x: torch.Tensor, v: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        groups: int, nonlin: bool, eps: float):
    B, C, H, W = x.shape
    assert C % groups == 0
    if nonlin:
        n = (x * v).sigmoid_().reshape(B, groups, -1)
        x = x.reshape(B, groups, -1)
        x = n / x.var(dim=-1, unbiased=False, keepdim=True).add_(eps).sqrt_()
        x = x.reshape(B, C, H, W)
    return x.mul_(weight).add_(bias)


class EvoNormSample2d(nn.Module):
    def __init__(self, num_features, nonlin=True, groups=8, eps=1e-5, jit=True):
        super(EvoNormSample2d, self).__init__()
        self.nonlin = nonlin
        self.groups = groups
        self.eps = eps
        self.jit = jit
        param_shape = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if nonlin:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.nonlin:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'

        if self.jit:
            return evo_sample_jit(
                x, self.v, self.weight, self.bias, self.groups, self.nonlin, self.eps)
        else:
            B, C, H, W = x.shape
            assert C % self.groups == 0
            if self.nonlin:
                n = (x * self.v).sigmoid().reshape(B, self.groups, -1)
                x = x.reshape(B, self.groups, -1)
                x = n / (x.std(dim=-1, unbiased=False, keepdim=True) + self.eps)
                x = x.reshape(B, C, H, W)
                return x.mul_(self.weight).add_(self.bias)
            else:
                return x.mul(self.weight).add_(self.bias)
