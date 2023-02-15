""" 'Fast' Normalization Functions

For GroupNorm and LayerNorm these functions bypass typical AMP upcast to float32.

Additionally, for LayerNorm, the APEX fused LN is used if available (which also does not upcast)

Hacked together by / Copyright 2022 Ross Wightman
"""
from typing import List, Optional

import torch
from torch.nn import functional as F

try:
    from apex.normalization.fused_layer_norm import fused_layer_norm_affine
    has_apex = True
except ImportError:
    has_apex = False

try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine, fused_rms_norm
    has_apex_rmsnorm = True
except ImportError:
    has_apex_rmsnorm = False


# fast (ie lower precision LN) can be disabled with this flag if issues crop up
_USE_FAST_NORM = False  # defaulting to False for now


def is_fast_norm():
    return _USE_FAST_NORM


def set_fast_norm(enable=True):
    global _USE_FAST_NORM
    _USE_FAST_NORM = enable


def fast_group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # currently cannot use is_autocast_enabled within torchscript
        return F.group_norm(x, num_groups, weight, bias, eps)

    if torch.is_autocast_enabled():
        # normally native AMP casts GN inputs to float32
        # here we use the low precision autocast dtype
        # FIXME what to do re CPU autocast?
        dt = torch.get_autocast_gpu_dtype()
        x, weight, bias = x.to(dt), weight.to(dt), bias.to(dt)

    with torch.cuda.amp.autocast(enabled=False):
        return F.group_norm(x, num_groups, weight, bias, eps)


def fast_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # currently cannot use is_autocast_enabled within torchscript
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    if has_apex:
        return fused_layer_norm_affine(x, weight, bias, normalized_shape, eps)

    if torch.is_autocast_enabled():
        # normally native AMP casts LN inputs to float32
        # apex LN does not, this is behaving like Apex
        dt = torch.get_autocast_gpu_dtype()
        # FIXME what to do re CPU autocast?
        x, weight, bias = x.to(dt), weight.to(dt), bias.to(dt)

    with torch.cuda.amp.autocast(enabled=False):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)


def rms_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    norm_ndim = len(normalized_shape)
    if torch.jit.is_scripting():
        # ndim = len(x.shape)
        # dims = list(range(ndim - norm_ndim, ndim))  # this doesn't work on pytorch <= 1.13.x
        # NOTE -ve dims cause torchscript to crash in some cases, out of options to work around
        assert norm_ndim == 1
        v = torch.var(x, dim=-1).unsqueeze(-1)  # ts crashes with -ve dim + keepdim=True
    else:
        dims = tuple(range(-1, -norm_ndim - 1, -1))
        v = torch.var(x, dim=dims, keepdim=True)
    x = x * torch.rsqrt(v + eps)
    if weight is not None:
        x = x * weight
    return x


def fast_rms_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # this must be by itself, cannot merge with has_apex_rmsnorm
        return rms_norm(x, normalized_shape, weight, eps)

    if has_apex_rmsnorm:
        if weight is None:
            return fused_rms_norm(x, normalized_shape, eps)
        else:
            return fused_rms_norm_affine(x, weight, normalized_shape, eps)

    # fallback
    return rms_norm(x, normalized_shape, weight, eps)
