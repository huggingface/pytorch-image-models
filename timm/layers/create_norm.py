""" Norm Layer Factory

Create norm modules by string (to mirror create_act and creat_norm-act fns)

Copyright 2022 Ross Wightman
"""
import types
import functools

import torch.nn as nn

from .norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d

_NORM_MAP = dict(
    batchnorm=nn.BatchNorm2d,
    batchnorm2d=nn.BatchNorm2d,
    batchnorm1d=nn.BatchNorm1d,
    groupnorm=GroupNorm,
    groupnorm1=GroupNorm1,
    layernorm=LayerNorm,
    layernorm2d=LayerNorm2d,
)
_NORM_TYPES = {m for n, m in _NORM_MAP.items()}


def create_norm_layer(layer_name, num_features, **kwargs):
    layer = get_norm_layer(layer_name)
    layer_instance = layer(num_features, **kwargs)
    return layer_instance


def get_norm_layer(norm_layer):
    assert isinstance(norm_layer, (type, str,  types.FunctionType, functools.partial))
    norm_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        layer_name = norm_layer.replace('_', '')
        norm_layer = _NORM_MAP.get(layer_name, None)
    elif norm_layer in _NORM_TYPES:
        norm_layer = norm_layer
    elif isinstance(norm_layer, types.FunctionType):
        # if function type, assume it is a lambda/fn that creates a norm layer
        norm_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower().replace('_', '')
        norm_layer = _NORM_MAP.get(type_name, None)
        assert norm_layer is not None, f"No equivalent norm layer for {type_name}"

    if norm_kwargs:
        norm_layer = functools.partial(norm_layer, **norm_kwargs)  # bind/rebind args
    return norm_layer
