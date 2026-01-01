""" Norm Layer Factory

Create norm modules by string (to mirror create_act and creat_norm-act fns)

Copyright 2022 Ross Wightman
"""
import functools
import inspect
import types
from typing import Type

import torch.nn as nn

from .norm import (
    GroupNorm,
    GroupNorm1,
    LayerNorm,
    LayerNorm2d,
    LayerNormFp32,
    LayerNorm2dFp32,
    RmsNorm,
    RmsNorm2d,
    RmsNormFp32,
    RmsNorm2dFp32,
    SimpleNorm,
    SimpleNorm2d,
    SimpleNormFp32,
    SimpleNorm2dFp32,
)
from torchvision.ops.misc import FrozenBatchNorm2d

_NORM_MAP = dict(
    batchnorm=nn.BatchNorm2d,
    batchnorm2d=nn.BatchNorm2d,
    batchnorm1d=nn.BatchNorm1d,
    groupnorm=GroupNorm,
    groupnorm1=GroupNorm1,
    layernorm=LayerNorm,
    layernorm2d=LayerNorm2d,
    layernormfp32=LayerNormFp32,
    layernorm2dfp32=LayerNorm2dFp32,
    rmsnorm=RmsNorm,
    rmsnorm2d=RmsNorm2d,
    rmsnormfp32=RmsNormFp32,
    rmsnorm2dfp32=RmsNorm2dFp32,
    simplenorm=SimpleNorm,
    simplenorm2d=SimpleNorm2d,
    simplenormfp32=SimpleNormFp32,
    simplenorm2dfp32=SimpleNorm2dFp32,
    frozenbatchnorm2d=FrozenBatchNorm2d,
)


def create_norm_layer(layer_name, **kwargs):
    layer = get_norm_layer(layer_name)
    # filter kwargs before creating layer
    params = inspect.signature(layer).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    layer_instance = layer(**kwargs)
    return layer_instance


def get_norm_layer(norm_layer):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    norm_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace('_', '').lower()
        norm_layer = _NORM_MAP[layer_name]
    else:
        norm_layer = norm_layer

    if norm_kwargs:
        norm_layer = functools.partial(norm_layer, **norm_kwargs)  # bind/rebind args
    return norm_layer
