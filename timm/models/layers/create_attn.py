""" Select AttentionFactory Method

Hacked together by Ross Wightman
"""
import torch
from .se import SEModule
from .eca import EcaModule, CecaModule


def create_attn(attn_type, channels, **kwargs):
    module_cls = None
    if attn_type is not None:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'eca':
                module_cls = CecaModule
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
