from .activations import *
from .activations_jit import *
from .activations_me import *
from .config import is_exportable, is_scriptable, is_no_jit


_ACT_FN_DEFAULT = dict(
    swish=swish,
    mish=mish,
    relu=F.relu,
    relu6=F.relu6,
    leaky_relu=F.leaky_relu,
    elu=F.elu,
    prelu=F.prelu,
    celu=F.celu,
    selu=F.selu,
    gelu=F.gelu,
    sigmoid=sigmoid,
    tanh=tanh,
    hard_sigmoid=hard_sigmoid,
    hard_swish=hard_swish,
    hard_mish=hard_mish,
)

_ACT_FN_JIT = dict(
    swish=swish_jit,
    mish=mish_jit,
    hard_sigmoid=hard_sigmoid_jit,
    hard_swish=hard_swish_jit,
    hard_mish=hard_mish_jit
)

_ACT_FN_ME = dict(
    swish=swish_me,
    mish=mish_me,
    hard_sigmoid=hard_sigmoid_me,
    hard_swish=hard_swish_me,
    hard_mish=hard_mish_me,
)

_ACT_LAYER_DEFAULT = dict(
    swish=Swish,
    mish=Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=HardSigmoid,
    hard_swish=HardSwish,
    hard_mish=HardMish,
)

_ACT_LAYER_JIT = dict(
    swish=SwishJit,
    mish=MishJit,
    hard_sigmoid=HardSigmoidJit,
    hard_swish=HardSwishJit,
    hard_mish=HardMishJit
)

_ACT_LAYER_ME = dict(
    swish=SwishMe,
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
    if not is_no_jit():
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
    if not is_no_jit():
        if name in _ACT_LAYER_JIT:
            return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name, inplace=False, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is not None:
        return act_layer(inplace=inplace, **kwargs)
    else:
        return None
