import types
import functools

import torch
import torch.nn as nn

from .evo_norm import EvoNormBatch2d, EvoNormSample2d
from .norm_act import BatchNormAct2d, GroupNormAct
from .inplace_abn import InplaceAbn

_NORM_ACT_TYPES = {BatchNormAct2d, GroupNormAct, EvoNormBatch2d, EvoNormSample2d, InplaceAbn}


def get_norm_act_layer(layer_class):
    layer_class = layer_class.replace('_', '').lower()
    if layer_class.startswith("batchnorm"):
        layer = BatchNormAct2d
    elif layer_class.startswith("groupnorm"):
        layer = GroupNormAct
    elif layer_class == "evonormbatch":
        layer = EvoNormBatch2d
    elif layer_class == "evonormsample":
        layer = EvoNormSample2d
    elif layer_class == "iabn" or layer_class == "inplaceabn":
        layer = InplaceAbn
    else:
        assert False, "Invalid norm_act layer (%s)" % layer_class
    return layer


def create_norm_act(layer_type, num_features, apply_act=True, jit=False, **kwargs):
    layer_parts = layer_type.split('-')  # e.g. batchnorm-leaky_relu
    assert len(layer_parts) in (1, 2)
    layer = get_norm_act_layer(layer_parts[0])
    #activation_class = layer_parts[1].lower() if len(layer_parts) > 1 else ''   # FIXME support string act selection?
    layer_instance = layer(num_features, apply_act=apply_act, **kwargs)
    if jit:
        layer_instance = torch.jit.script(layer_instance)
    return layer_instance


def convert_norm_act_type(norm_layer, act_layer, norm_kwargs=None):
    assert isinstance(norm_layer, (type, str,  types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_args = norm_kwargs.copy() if norm_kwargs else {}
    if isinstance(norm_layer, str):
        norm_act_layer = get_norm_act_layer(norm_layer)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  (types.FunctionType, functools.partial)):
        # assuming this is a lambda/fn/bound partial that creates norm_act layer
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnorm'):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith('groupnorm'):
            norm_act_layer = GroupNormAct
        else:
            assert False, f"No equivalent norm_act layer for {type_name}"
        # Must pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # Newer models will use `apply_act` and likely have `act_layer` arg bound to relevant NormAct types.
        norm_act_args.update(dict(act_layer=act_layer))
    return norm_act_layer, norm_act_args
