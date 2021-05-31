""" Attention Factory

Hacked together by / Copyright 2021 Ross Wightman
"""
import torch
from functools import partial

from .bottleneck_attn import BottleneckAttn
from .cbam import CbamModule, LightCbamModule
from .eca import EcaModule, CecaModule
from .gather_excite import GatherExcite
from .global_context import GlobalContext
from .halo_attn import HaloAttn
from .involution import Involution
from .lambda_layer import LambdaLayer
from .non_local_attn import NonLocalAttn, BatNonLocalAttn
from .selective_kernel import SelectiveKernel
from .split_attn import SplitAttn
from .squeeze_excite import SEModule, EffectiveSEModule
from .swin_attn import WindowAttention


def get_attn(attn_type):
    if isinstance(attn_type, torch.nn.Module):
        return attn_type
    module_cls = None
    if attn_type is not None:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            # Lightweight attention modules (channel and/or coarse spatial).
            # Typically added to existing network architecture blocks in addition to existing convolutions.
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ecam':
                module_cls = partial(EcaModule, use_mlp=True)
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'ge':
                module_cls = GatherExcite
            elif attn_type == 'gc':
                module_cls = GlobalContext
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule

            # Attention / attention-like modules w/ significant params
            # Typically replace some of the existing workhorse convs in a network architecture.
            # All of these accept a stride argument and can spatially downsample the input.
            elif attn_type == 'sk':
                module_cls = SelectiveKernel
            elif attn_type == 'splat':
                module_cls = SplitAttn

            # Self-attention / attention-like modules w/ significant compute and/or params
            # Typically replace some of the existing workhorse convs in a network architecture.
            # All of these accept a stride argument and can spatially downsample the input.
            elif attn_type == 'lambda':
                return LambdaLayer
            elif attn_type == 'bottleneck':
                return BottleneckAttn
            elif attn_type == 'halo':
                return HaloAttn
            elif attn_type == 'swin':
                return WindowAttention
            elif attn_type == 'involution':
                return Involution
            elif attn_type == 'nl':
                module_cls = NonLocalAttn
            elif attn_type == 'bat':
                module_cls = BatNonLocalAttn

            # Woops!
            else:
                assert False, "Invalid attn module (%s)" % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    return module_cls


def create_attn(attn_type, channels, **kwargs):
    module_cls = get_attn(attn_type)
    if module_cls is not None:
        # NOTE: it's expected the first (positional) argument of all attention layers is the # input channels
        return module_cls(channels, **kwargs)
    return None
