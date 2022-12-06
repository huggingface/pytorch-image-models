""" Normalization layers and wrappers

Norm layer definitions that support fast norm and consistent channel arg order (always first arg).

Hacked together by / Copyright 2022 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_norm import is_fast_norm, fast_group_norm, fast_layer_norm


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        # NOTE num_channels is swapped to first arg for consistency in swapping norm layers with BN
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)
        self.fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x):
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class GroupNorm1(nn.GroupNorm):
    """ Group Normalization with 1 group.
    Input: tensor in shape [B, C, *]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
        self.fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class LayerNorm(nn.LayerNorm):
    """ LayerNorm w/ fast norm option
    """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


@torch.jit.script
def _layer_norm_cf(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
    x = (x - u) * torch.rsqrt(s + eps)
    x = x * weight[:, None, None] + bias[:, None, None]
    return x


def _layer_norm_cf_sqm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    u = x.mean(dim=1, keepdim=True)
    s = ((x * x).mean(dim=1, keepdim=True) - (u * u)).clamp(0)
    x = (x - u) * torch.rsqrt(s + eps)
    x = x * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    return x


class LayerNormExp2d(nn.LayerNorm):
    """ LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).

    Experimental implementation w/ manual norm for tensors non-contiguous tensors.

    This improves throughput in some scenarios (tested on Ampere GPU), esp w/ channels_last
    layout. However, benefits are not always clear and can perform worse on other GPUs.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__(num_channels, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            x = F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            x = _layer_norm_cf(x, self.weight, self.bias, self.eps)
        return x
