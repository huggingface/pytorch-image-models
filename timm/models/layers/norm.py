""" Normalization layers and wrappers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        # NOTE num_channels is swapped to first arg for consistency in swapping norm layers with BN
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class GroupNorm1(nn.GroupNorm):
    """ Group Normalization with 1 group.
    Input: tensor in shape [B, C, *]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
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
