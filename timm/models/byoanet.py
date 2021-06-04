""" Bring-Your-Own-Attention Network

A flexible network w/ dataclass based config for stacking NN blocks including
self-attention (or similar) layers.

Currently used to implement experimential variants of:
  * Bottleneck Transformers
  * Lambda ResNets
  * HaloNets

Consider all of the models definitions here as experimental WIP and likely to change.

Hacked together by / copyright Ross Wightman, 2021.
"""
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .byobnet import ByoBlockCfg, ByoModelCfg, ByobNet, interleave_blocks
from .helpers import build_model_with_cfg
from .registry import register_model

__all__ = []


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.conv', 'classifier': 'head.fc',
        'fixed_input_size': False, 'min_input_size': (3, 224, 224),
        **kwargs
    }


default_cfgs = {
    # GPU-Efficient (ResNet) weights
    'botnet26t_256': _cfg(url='', fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'botnet50ts_256': _cfg(url='', fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'eca_botnext26ts_256': _cfg(url='', fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),

    'halonet_h1': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'halonet_h1_c4c5': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'halonet26t': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'halonet50ts': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'eca_halonext26ts': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),

    'lambda_resnet26t': _cfg(url='', min_input_size=(3, 128, 128), input_size=(3, 256, 256), pool_size=(8, 8)),
    'lambda_resnet50t': _cfg(url='', min_input_size=(3, 128, 128)),
    'eca_lambda_resnext26ts': _cfg(url='', min_input_size=(3, 128, 128), input_size=(3, 256, 256), pool_size=(8, 8)),

    'swinnet26t_256': _cfg(url='', fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'swinnet50ts_256': _cfg(url='', fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'eca_swinnext26ts_256': _cfg(url='', fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),

    'rednet26t': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8)),
    'rednet50ts': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8)),
}


model_cfgs = dict(

    botnet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        fixed_input_size=True,
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),
    botnet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=6, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=3, c=2048, s=1, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        num_features=0,
        fixed_input_size=True,
        act_layer='silu',
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),
    eca_botnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=16, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=16, br=0.25),
            ByoBlockCfg(type='self_attn', d=3, c=2048, s=2, gs=16, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        fixed_input_size=True,
        act_layer='silu',
        attn_layer='eca',
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),

    halonet_h1=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='self_attn', d=3, c=64, s=1, gs=0, br=1.0),
            ByoBlockCfg(type='self_attn', d=3, c=128, s=2, gs=0, br=1.0),
            ByoBlockCfg(type='self_attn', d=10, c=256, s=2, gs=0, br=1.0),
            ByoBlockCfg(type='self_attn', d=3, c=512, s=2, gs=0, br=1.0),
        ),
        stem_chs=64,
        stem_type='7x7',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3),
    ),
    halonet_h1_c4c5=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=64, s=1, gs=0, br=1.0),
            ByoBlockCfg(type='bottle', d=3, c=128, s=2, gs=0, br=1.0),
            ByoBlockCfg(type='self_attn', d=10, c=256, s=2, gs=0, br=1.0),
            ByoBlockCfg(type='self_attn', d=3, c=512, s=2, gs=0, br=1.0),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3),
    ),
    halonet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=2)  # intended for 256x256 res
    ),
    halonet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=6, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        act_layer='silu',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=2)
    ),
    eca_halonext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=16, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=16, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=16, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        act_layer='silu',
        attn_layer='eca',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=2)  # intended for 256x256 res
    ),

    lambda_resnet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='lambda',
        self_attn_kwargs=dict()
    ),
    lambda_resnet50t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=3, d=6, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='lambda',
        self_attn_kwargs=dict()
    ),
    eca_lambda_resnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=16, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=16, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=16, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        act_layer='silu',
        attn_layer='eca',
        self_attn_layer='lambda',
        self_attn_kwargs=dict()
    ),

    swinnet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        fixed_input_size=True,
        self_attn_layer='swin',
        self_attn_kwargs=dict(win_size=8)
    ),
    swinnet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=4, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        fixed_input_size=True,
        act_layer='silu',
        self_attn_layer='swin',
        self_attn_kwargs=dict(win_size=8)
    ),
    eca_swinnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=512, s=2, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=16, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=16, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        fixed_input_size=True,
        act_layer='silu',
        attn_layer='eca',
        self_attn_layer='swin',
        self_attn_kwargs=dict(win_size=8)
    ),


    rednet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='self_attn', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=512, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',  # FIXME RedNet uses involution in middle of stem
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='involution',
        self_attn_kwargs=dict()
    ),
    rednet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='self_attn', d=3, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=4, c=512, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        act_layer='silu',
        self_attn_layer='involution',
        self_attn_kwargs=dict()
    ),
)


def _create_byoanet(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)


@register_model
def botnet26t_256(pretrained=False, **kwargs):
    """ Bottleneck Transformer w/ ResNet26-T backbone. Bottleneck attn in final stage.
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('botnet26t_256', 'botnet26t', pretrained=pretrained, **kwargs)


@register_model
def botnet50ts_256(pretrained=False, **kwargs):
    """ Bottleneck Transformer w/ ResNet50-T backbone. Bottleneck attn in final stage.
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('botnet50ts_256', 'botnet50ts', pretrained=pretrained, **kwargs)


@register_model
def eca_botnext26ts_256(pretrained=False, **kwargs):
    """ Bottleneck Transformer w/ ResNet26-T backbone. Bottleneck attn in final stage.
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('eca_botnext26ts_256', 'eca_botnext26ts', pretrained=pretrained, **kwargs)


@register_model
def halonet_h1(pretrained=False, **kwargs):
    """ HaloNet-H1. Halo attention in all stages as per the paper.

    This runs very slowly, param count lower than paper --> something is wrong.
    """
    return _create_byoanet('halonet_h1', pretrained=pretrained, **kwargs)


@register_model
def halonet_h1_c4c5(pretrained=False, **kwargs):
    """ HaloNet-H1 config w/ attention in last two stages.
    """
    return _create_byoanet('halonet_h1_c4c5', pretrained=pretrained, **kwargs)


@register_model
def halonet26t(pretrained=False, **kwargs):
    """ HaloNet w/ a ResNet26-t backbone, Hallo attention in final stage
    """
    return _create_byoanet('halonet26t', pretrained=pretrained, **kwargs)


@register_model
def halonet50ts(pretrained=False, **kwargs):
    """ HaloNet w/ a ResNet50-t backbone, Hallo attention in final stage
    """
    return _create_byoanet('halonet50ts', pretrained=pretrained, **kwargs)


@register_model
def eca_halonext26ts(pretrained=False, **kwargs):
    """ HaloNet w/ a ResNet26-t backbone, Hallo attention in final stage
    """
    return _create_byoanet('eca_halonext26ts', pretrained=pretrained, **kwargs)


@register_model
def lambda_resnet26t(pretrained=False, **kwargs):
    """ Lambda-ResNet-26T. Lambda layers in one C4 stage and all C5.
    """
    return _create_byoanet('lambda_resnet26t', pretrained=pretrained, **kwargs)


@register_model
def lambda_resnet50t(pretrained=False, **kwargs):
    """ Lambda-ResNet-50T. Lambda layers in one C4 stage and all C5.
    """
    return _create_byoanet('lambda_resnet50t', pretrained=pretrained, **kwargs)


@register_model
def eca_lambda_resnext26ts(pretrained=False, **kwargs):
    """ Lambda-ResNet-26T. Lambda layers in one C4 stage and all C5.
    """
    return _create_byoanet('eca_lambda_resnext26ts', pretrained=pretrained, **kwargs)


@register_model
def swinnet26t_256(pretrained=False, **kwargs):
    """
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('swinnet26t_256', 'swinnet26t', pretrained=pretrained, **kwargs)


@register_model
def swinnet50ts_256(pretrained=False, **kwargs):
    """
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('swinnet50ts_256', 'swinnet50ts', pretrained=pretrained, **kwargs)


@register_model
def eca_swinnext26ts_256(pretrained=False, **kwargs):
    """
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('eca_swinnext26ts_256', 'eca_swinnext26ts', pretrained=pretrained, **kwargs)


@register_model
def rednet26t(pretrained=False, **kwargs):
    """
    """
    return _create_byoanet('rednet26t', pretrained=pretrained, **kwargs)


@register_model
def rednet50ts(pretrained=False, **kwargs):
    """
    """
    return _create_byoanet('rednet50ts', pretrained=pretrained, **kwargs)
