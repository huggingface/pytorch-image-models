""" Model / Layer Config singleton state
"""
import os
import warnings
from typing import Any, Optional

import torch

__all__ = [
    'is_exportable', 'is_scriptable', 'is_no_jit', 'use_fused_attn',
    'set_exportable', 'set_scriptable', 'set_no_jit', 'set_layer_config', 'set_fused_attn',
    'set_reentrant_ckpt', 'use_reentrant_ckpt'
]

# Set to True if prefer to have layers with no jit optimization (includes activations)
_NO_JIT = False

# Set to True if prefer to have activation layers with no jit optimization
# NOTE not currently used as no difference between no_jit and no_activation jit as only layers obeying
# the jit flags so far are activations. This will change as more layers are updated and/or added.
_NO_ACTIVATION_JIT = False

# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False

# Set to True if wanting to use torch.jit.script on a model
_SCRIPTABLE = False


# use torch.scaled_dot_product_attention where possible
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if 'TIMM_FUSED_ATTN' in os.environ:
    _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
else:
    _USE_FUSED_ATTN = 1  # 0 == off, 1 == on (for tested use), 2 == on (for experimental use)


if 'TIMM_REENTRANT_CKPT' in os.environ:
    _USE_REENTRANT_CKPT = bool(os.environ['TIMM_REENTRANT_CKPT'])
else:
    _USE_REENTRANT_CKPT = False  # defaults to disabled (off)


def is_no_jit():
    return _NO_JIT


class set_no_jit:
    def __init__(self, mode: bool) -> None:
        global _NO_JIT
        self.prev = _NO_JIT
        _NO_JIT = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _NO_JIT
        _NO_JIT = self.prev
        return False


def is_exportable():
    return _EXPORTABLE


class set_exportable:
    def __init__(self, mode: bool) -> None:
        global _EXPORTABLE
        self.prev = _EXPORTABLE
        _EXPORTABLE = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _EXPORTABLE
        _EXPORTABLE = self.prev
        return False


def is_scriptable():
    return _SCRIPTABLE


class set_scriptable:
    def __init__(self, mode: bool) -> None:
        global _SCRIPTABLE
        self.prev = _SCRIPTABLE
        _SCRIPTABLE = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _SCRIPTABLE
        _SCRIPTABLE = self.prev
        return False


class set_layer_config:
    """ Layer config context manager that allows setting all layer config flags at once.
    If a flag arg is None, it will not change the current value.
    """
    def __init__(
            self,
            scriptable: Optional[bool] = None,
            exportable: Optional[bool] = None,
            no_jit: Optional[bool] = None,
            no_activation_jit: Optional[bool] = None):
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        self.prev = _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT
        if scriptable is not None:
            _SCRIPTABLE = scriptable
        if exportable is not None:
            _EXPORTABLE = exportable
        if no_jit is not None:
            _NO_JIT = no_jit
        if no_activation_jit is not None:
            _NO_ACTIVATION_JIT = no_activation_jit

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> bool:
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT = self.prev
        return False


def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0


def set_fused_attn(enable: bool = True, experimental: bool = False):
    global _USE_FUSED_ATTN
    if not _HAS_FUSED_ATTN:
        warnings.warn('This version of pytorch does not have F.scaled_dot_product_attention, fused_attn flag ignored.')
        return
    if experimental and enable:
        _USE_FUSED_ATTN = 2
    elif enable:
        _USE_FUSED_ATTN = 1
    else:
        _USE_FUSED_ATTN = 0


def use_reentrant_ckpt() -> bool:
    return _USE_REENTRANT_CKPT


def set_reentrant_ckpt(enable: bool = True):
    global _USE_REENTRANT_CKPT
    _USE_REENTRANT_CKPT = enable
