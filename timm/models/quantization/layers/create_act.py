""" Activation Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .activations import *
from .activations_jit import *
from .activations_me import *
from .config import is_exportable, is_scriptable, is_no_jit

_ACT_LAYER_DEFAULT = dict(
    swish=Swish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    sigmoid=Sigmoid,
    hard_sigmoid=HardSigmoid,
    hard_swish=HardSwish,
)



def get_act_layer(name='relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
        
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name, inplace=False, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is not None:
        return act_layer(inplace=inplace, **kwargs)
    else:
        return None
