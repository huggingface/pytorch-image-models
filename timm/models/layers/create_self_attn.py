from .bottleneck_attn import BottleneckAttn
from .halo_attn import HaloAttn
from .lambda_layer import LambdaLayer


def get_self_attn(attn_type):
    if attn_type == 'bottleneck':
        return BottleneckAttn
    elif attn_type == 'halo':
        return HaloAttn
    elif attn_type == 'lambda':
        return LambdaLayer


def create_self_attn(attn_type, dim, stride=1, **kwargs):
    attn_fn = get_self_attn(attn_type)
    return attn_fn(dim, stride=stride, **kwargs)
