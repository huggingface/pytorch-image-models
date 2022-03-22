""" NormAct (Normalizaiton + Activation Layer) Factory

Create norm + act combo modules that attempt to be backwards compatible with separate norm + act
isntances in models. Where these are used it will be possible to swap separate BN + act layers with
combined modules like IABN or EvoNorms.

Hacked together by / Copyright 2020 Ross Wightman
"""
import types
import functools

from .evo_norm import *
from .filter_response_norm import FilterResponseNormAct2d, FilterResponseNormTlu2d
from .norm_act import BatchNormAct2d, GroupNormAct, LayerNormAct, LayerNormAct2d
from .inplace_abn import InplaceAbn

_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
    batchnorm2d=BatchNormAct2d,
    groupnorm=GroupNormAct,
    layernorm=LayerNormAct,
    layernorm2d=LayerNormAct2d,
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
)
_NORM_ACT_TYPES = {m for n, m in _NORM_ACT_MAP.items()}
# has act_layer arg to define act type
_NORM_ACT_REQUIRES_ARG = {
    BatchNormAct2d, GroupNormAct, LayerNormAct, LayerNormAct2d, FilterResponseNormAct2d, InplaceAbn}


def create_norm_act_layer(layer_name, num_features, act_layer=None, apply_act=True, jit=False, **kwargs):
    layer = get_norm_act_layer(layer_name, act_layer=act_layer)
    layer_instance = layer(num_features, apply_act=apply_act, **kwargs)
    if jit:
        layer_instance = torch.jit.script(layer_instance)
    return layer_instance


def get_norm_act_layer(norm_layer, act_layer=None):
    assert isinstance(norm_layer, (type, str,  types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP.get(layer_name, None)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  types.FunctionType):
        # if function type, must be a lambda/fn that creates a norm_act layer
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnorm'):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith('groupnorm'):
            norm_act_layer = GroupNormAct
        elif type_name.startswith('layernorm2d'):
            norm_act_layer = LayerNormAct2d
        elif type_name.startswith('layernorm'):
            norm_act_layer = LayerNormAct
        else:
            assert False, f"No equivalent norm_act layer for {type_name}"

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args
    return norm_act_layer
