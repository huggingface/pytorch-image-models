""" Filter Response Norm in PyTorch

Based on `Filter Response Normalization Layer` - https://arxiv.org/abs/1911.09737

Hacked together by / Copyright 2021 Ross Wightman
"""
import torch
import torch.nn as nn

from .create_act import create_act_layer
from .trace_utils import _assert


def inv_instance_rms(x, eps: float = 1e-5):
    rms = x.square().float().mean(dim=(2, 3), keepdim=True).add(eps).rsqrt().to(x.dtype)
    return rms.expand(x.shape)


class FilterResponseNormTlu2d(nn.Module):
    def __init__(self, num_features, apply_act=True, eps=1e-5, rms=True, **_):
        super(FilterResponseNormTlu2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.rms = rms
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.tau = nn.Parameter(torch.zeros(num_features)) if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.tau is not None:
            nn.init.zeros_(self.tau)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)
        return torch.maximum(x, self.tau.reshape(v_shape).to(dtype=x_dtype)) if self.tau is not None else x


class FilterResponseNormAct2d(nn.Module):
    def __init__(self, num_features, apply_act=True, act_layer=nn.ReLU, inplace=None, rms=True, eps=1e-5, **_):
        super(FilterResponseNormAct2d, self).__init__()
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer, inplace=inplace)
        else:
            self.act = nn.Identity()
        self.rms = rms
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
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)
        return self.act(x)
