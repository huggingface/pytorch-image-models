""" NormAct (Normalization + Activation Layer) Factory

Create norm + act combo modules that attempt to be backwards compatible with separate norm + act
instances in models. Where these are used it will be possible to swap separate BN + act layers with
combined modules like IABN or EvoNorms.

Hacked together by / Copyright 2020 Ross Wightman
"""
import types
import functools
from typing import Optional

from .evo_norm import *
from .filter_response_norm import FilterResponseNormAct2d, FilterResponseNormTlu2d
from .norm_act import (
    BatchNormAct2d,
    GroupNormAct,
    GroupNorm1Act,
    LayerNormAct,
    LayerNormActFp32,
    LayerNormAct2d,
    LayerNormAct2dFp32,
    RmsNormAct,
    RmsNormActFp32,
    RmsNormAct2d,
    RmsNormAct2dFp32,
)
from .inplace_abn import InplaceAbn
from .typing import LayerType

_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
    batchnorm2d=BatchNormAct2d,
    groupnorm=GroupNormAct,
    groupnorm1=GroupNorm1Act,
    layernorm=LayerNormAct,
    layernorm2d=LayerNormAct2d,
    layernormfp32=LayerNormActFp32,
    layernorm2dfp32=LayerNormAct2dFp32,
    evonormb0=EvoNorm2dB0,
    evonormb1=EvoNorm2dB1,
    evonormb2=EvoNorm2dB2,
    evonorms0=EvoNorm2dS0,
    evonorms0a=EvoNorm2dS0a,
    evonorms1=EvoNorm2dS1,
    evonorms1a=EvoNorm2dS1a,
    evonorms2=EvoNorm2dS2,
    evonorms2a=EvoNorm2dS2a,
    frn=FilterResponseNormAct2d,
    frntlu=FilterResponseNormTlu2d,
    inplaceabn=InplaceAbn,
    iabn=InplaceAbn,
    rmsnorm=RmsNormAct,
    rmsnorm2d=RmsNormAct2d,
    rmsnormfp32=RmsNormActFp32,
    rmsnorm2dfp32=RmsNormAct2dFp32,
)
_NORM_ACT_TYPES = {m for n, m in _NORM_ACT_MAP.items()}
# Reverse map from base norm layer names to norm+act layer classes
_NORM_TO_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
    batchnorm2d=BatchNormAct2d,
    groupnorm=GroupNormAct,
    groupnorm1=GroupNorm1Act,
    layernorm=LayerNormAct,
    layernorm2d=LayerNormAct2d,
    layernormfp32=LayerNormActFp32,
    layernorm2dfp32=LayerNormAct2dFp32,
    rmsnorm=RmsNormAct,
    rmsnorm2d=RmsNormAct2d,
    rmsnormfp32=RmsNormActFp32,
    rmsnorm2dfp32=RmsNormAct2dFp32,
)
# has act_layer arg to define act type
_NORM_ACT_REQUIRES_ARG = {
    BatchNormAct2d,
    GroupNormAct,
    GroupNorm1Act,
    LayerNormAct,
    LayerNormAct2d,
    LayerNormActFp32,
    LayerNormAct2dFp32,
    FilterResponseNormAct2d,
    InplaceAbn,
    RmsNormAct,
    RmsNormAct2d,
    RmsNormActFp32,
    RmsNormAct2dFp32,
}


def create_norm_act_layer(
        layer_name: LayerType,
        num_features: int,
        act_layer: Optional[LayerType] = None,
        apply_act: bool = True,
        jit: bool = False,
        **kwargs,
):
    layer = get_norm_act_layer(layer_name, act_layer=act_layer)
    layer_instance = layer(num_features, apply_act=apply_act, **kwargs)
    if jit:
        layer_instance = torch.jit.script(layer_instance)
    return layer_instance


def get_norm_act_layer(
        norm_layer: LayerType,
        act_layer: Optional[LayerType] = None,
):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str,  types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP[layer_name]
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  types.FunctionType):
        # if function type, must be a lambda/fn that creates a norm_act layer
        norm_act_layer = norm_layer
    else:
        # Use reverse map to find the corresponding norm+act layer
        type_name = norm_layer.__name__.lower()
        norm_act_layer = _NORM_TO_NORM_ACT_MAP.get(type_name, None)
        assert norm_act_layer is not None, f"No equivalent norm_act layer for {type_name}"

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args

    return norm_act_layer
