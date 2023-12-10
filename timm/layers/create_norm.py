""" Norm Layer Factory

Create norm modules by string (to mirror create_act and creat_norm-act fns)

Copyright 2022 Ross Wightman
"""
import functools
import types
from typing import Type

import torch.nn as nn

from .norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d, RmsNorm
from torchvision.ops.misc import FrozenBatchNorm2d

_NORM_MAP = dict(
    batchnorm=nn.BatchNorm2d,
    batchnorm2d=nn.BatchNorm2d,
    batchnorm1d=nn.BatchNorm1d,
    groupnorm=GroupNorm,
    groupnorm1=GroupNorm1,
    layernorm=LayerNorm,
    layernorm2d=LayerNorm2d,
    rmsnorm=RmsNorm,
    frozenbatchnorm2d=FrozenBatchNorm2d,
)
_NORM_TYPES = {m for n, m in _NORM_MAP.items()}


def create_norm_layer(layer_name, num_features, **kwargs):
    layer = get_norm_layer(layer_name)
    layer_instance = layer(num_features, **kwargs)
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
        layer_name = norm_layer.replace('_', '')
        norm_layer = _NORM_MAP[layer_name]
    else:
        norm_layer = norm_layer

    if norm_kwargs:
        norm_layer = functools.partial(norm_layer, **norm_kwargs)  # bind/rebind args
    return norm_layer
