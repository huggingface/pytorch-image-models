""" Activation Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .activations import *
from .activations_jit import *
from .activations_me import *
from .config import is_exportable, is_scriptable, is_no_jit

# PyTorch has an optimized, native 'silu' (aka 'swish') operator as of PyTorch 1.7. This code
# will use native version if present. Eventually, the custom Swish layers will be removed
# and only native 'silu' will be used.
_has_silu = 'silu' in dir(torch.nn.functional)

_ACT_FN_DEFAULT = dict(
    silu=F.silu if _has_silu else swish,
    swish=F.silu if _has_silu else swish,
    mish=mish,
    relu=F.relu,
    relu6=F.relu6,
    leaky_relu=F.leaky_relu,
    elu=F.elu,
    celu=F.celu,
    selu=F.selu,
    gelu=gelu,
    sigmoid=sigmoid,
    tanh=tanh,
    hard_sigmoid=hard_sigmoid,
    hard_swish=hard_swish,
    hard_mish=hard_mish,
)

_ACT_FN_JIT = dict(
    silu=F.silu if _has_silu else swish_jit,
    swish=F.silu if _has_silu else swish_jit,
    mish=mish_jit,
    hard_sigmoid=hard_sigmoid_jit,
    hard_swish=hard_swish_jit,
    hard_mish=hard_mish_jit
)

_ACT_FN_ME = dict(
    silu=F.silu if _has_silu else swish_me,
    swish=F.silu if _has_silu else swish_me,
    mish=mish_me,
    hard_sigmoid=hard_sigmoid_me,
    hard_swish=hard_swish_me,
    hard_mish=hard_mish_me,
)

_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU if _has_silu else Swish,
    swish=nn.SiLU if _has_silu else Swish,
    mish=Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=GELU,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=HardSigmoid,
    hard_swish=HardSwish,
    hard_mish=HardMish,
)

_ACT_LAYER_JIT = dict(
    silu=nn.SiLU if _has_silu else SwishJit,
    swish=nn.SiLU if _has_silu else SwishJit,
    mish=MishJit,
    hard_sigmoid=HardSigmoidJit,
    hard_swish=HardSwishJit,
    hard_mish=HardMishJit
)

_ACT_LAYER_ME = dict(
    silu=nn.SiLU if _has_silu else SwishMe,
    swish=nn.SiLU if _has_silu else SwishMe,
    mish=MishMe,
    hard_sigmoid=HardSigmoidMe,
    hard_swish=HardSwishMe,
    hard_mish=HardMishMe,
)


def get_act_fn(name='relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        # If not exporting or scripting the model, first look for a memory-efficient version with
        # custom autograd, then fallback
        if name in _ACT_FN_ME:
            return _ACT_FN_ME[name]
    if is_exportable() and name in ('silu', 'swish'):
        # FIXME PyTorch SiLU doesn't ONNX export, this is a temp hack
        return swish
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_FN_JIT:
            return _ACT_FN_JIT[name]
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name='relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
    if is_exportable() and name in ('silu', 'swish'):
        # FIXME PyTorch SiLU doesn't ONNX export, this is a temp hack
        return Swish
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_LAYER_JIT:
            return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name, inplace=False, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is not None:
        return act_layer(inplace=inplace, **kwargs)
    else:
        return None
