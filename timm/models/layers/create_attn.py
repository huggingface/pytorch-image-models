""" Select AttentionFactory Method

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from .se import SEModule, EffectiveSEModule
from .eca import EcaModule, CecaModule
from .cbam import CbamModule, LightCbamModule


def create_attn(attn_type, channels, **kwargs):
    module_cls = None
    if attn_type is not None:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule
            else:
                assert False, "Invalid attn module (%s)" % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    if module_cls is not None:
        return module_cls(channels, **kwargs)
    return None
