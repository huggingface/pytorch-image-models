""" Bring-Your-Own-Attention Network

A flexible network w/ dataclass based config for stacking NN blocks including
self-attention (or similar) layers.

Currently used to implement experimental variants of:
  * Bottleneck Transformers
  * Lambda ResNets
  * HaloNets

Consider all of the models definitions here as experimental WIP and likely to change.

Hacked together by / copyright Ross Wightman, 2021.
"""
from typing import Any, Dict, Optional

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs
from .byobnet import ByoBlockCfg, ByoModelCfg, ByobNet, interleave_blocks

__all__ = []


model_cfgs = dict(

    botnet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        fixed_input_size=True,
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),
    sebotnet33ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=[2], d=3, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=[2], d=3, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg('self_attn', d=2, c=1536, s=2, gs=0, br=0.333),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        act_layer='silu',
        num_features=1280,
        attn_layer='se',
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),
    botnet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=4, d=4, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=6, c=1024, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        fixed_input_size=True,
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),
    eca_botnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=16, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=16, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=16, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        fixed_input_size=True,
        act_layer='silu',
        attn_layer='eca',
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict(dim_head=16)
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

        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3),
    ),
    halonet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=2)
    ),
    sehalonet33ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=[2], d=3, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=[2], d=3, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg('self_attn', d=2, c=1536, s=2, gs=0, br=0.333),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        act_layer='silu',
        num_features=1280,
        attn_layer='se',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3)
    ),
    halonet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(
                types=('bottle', 'self_attn'), every=4, d=4, c=512, s=2, gs=0, br=0.25,
                self_attn_layer='halo', self_attn_kwargs=dict(block_size=8, halo_size=3, num_heads=4)),
            interleave_blocks(types=('bottle', 'self_attn'), d=6, c=1024, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3)
    ),
    eca_halonext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=16, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=16, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=16, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        attn_layer='eca',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=2, dim_head=16)
    ),

    lambda_resnet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        self_attn_layer='lambda',
        self_attn_kwargs=dict(r=9)
    ),
    lambda_resnet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=4, d=4, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=6, c=1024, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        self_attn_layer='lambda',
        self_attn_kwargs=dict(r=9)
    ),
    lambda_resnet26rpt_256=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        self_attn_layer='lambda',
        self_attn_kwargs=dict(r=None)
    ),

    # experimental
    haloregnetz_b=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=48, s=2, gs=16, br=3),
            ByoBlockCfg(type='bottle', d=6, c=96, s=2, gs=16, br=3),
            interleave_blocks(types=('bottle', 'self_attn'), every=3, d=12, c=192, s=2, gs=16, br=3),
            ByoBlockCfg('self_attn', d=2, c=288, s=2, gs=16, br=3),
        ),
        stem_chs=32,
        stem_pool='',
        downsample='',
        num_features=1536,
        act_layer='silu',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=7, halo_size=2, qk_ratio=0.33)
    ),

    # experimental
    lamhalobotnet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(
                types=('bottle', 'self_attn'), d=4, c=512, s=2, gs=0, br=0.25,
                self_attn_layer='lambda', self_attn_kwargs=dict(r=13)),
            interleave_blocks(
                types=('bottle', 'self_attn'), d=6, c=1024, s=2, gs=0, br=0.25,
                self_attn_layer='halo', self_attn_kwargs=dict(halo_size=3)),
            interleave_blocks(
                types=('bottle', 'self_attn'), d=3, c=2048, s=2, gs=0, br=0.25,
                self_attn_layer='bottleneck', self_attn_kwargs=dict()),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        act_layer='silu',
    ),
    halo2botnet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(
                types=('bottle', 'self_attn'), d=4, c=512, s=2, gs=0, br=0.25,
                self_attn_layer='halo', self_attn_kwargs=dict(halo_size=3)),
            interleave_blocks(
                types=('bottle', 'self_attn'), d=6, c=1024, s=2, gs=0, br=0.25,
                self_attn_layer='halo', self_attn_kwargs=dict(halo_size=3)),
            interleave_blocks(
                types=('bottle', 'self_attn'), d=3, c=2048, s=2, gs=0, br=0.25,
                self_attn_layer='bottleneck', self_attn_kwargs=dict()),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        act_layer='silu',
    ),
)


def _create_byoanet(variant: str, cfg_variant: Optional[str] = None, pretrained: bool = False, **kwargs) -> ByobNet:
    """Create a Bring-Your-Own-Attention network model.

    Args:
        variant: Model variant name.
        cfg_variant: Config variant name if different from model variant.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        Instantiated ByobNet model.
    """
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    """Generate default model configuration.

    Args:
        url: URL for pretrained weights.
        **kwargs: Override default configuration values.

    Returns:
        Model configuration dictionary.
    """
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.conv', 'classifier': 'head.fc',
        'fixed_input_size': False, 'min_input_size': (3, 224, 224),
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # GPU-Efficient (ResNet) weights
    'botnet26t_256.c1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/botnet26t_c1_256-167a0e9f.pth',
        hf_hub_id='timm/',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'sebotnet33ts_256.a1h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/sebotnet33ts_a1h2_256-957e3c3e.pth',
        hf_hub_id='timm/',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.94),
    'botnet50ts_256.untrained': _cfg(
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'eca_botnext26ts_256.c1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/eca_botnext26ts_c_256-95a898f6.pth',
        hf_hub_id='timm/',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),

    'halonet_h1.untrained': _cfg(input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'halonet26t.a1h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/halonet26t_a1h_256-3083328c.pth',
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'sehalonet33ts.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/sehalonet33ts_256-87e053f9.pth',
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256), crop_pct=0.94),
    'halonet50ts.a1h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/halonet50ts_a1h2_256-f3a3daee.pth',
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256), crop_pct=0.94),
    'eca_halonext26ts.c1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/eca_halonext26ts_c_256-06906299.pth',
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256), crop_pct=0.94),

    'lambda_resnet26t.c1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/lambda_resnet26t_c_256-e5a5c857.pth',
        hf_hub_id='timm/',
        min_input_size=(3, 128, 128), input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.94),
    'lambda_resnet50ts.a1h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/lambda_resnet50ts_a1h_256-b87370f7.pth',
        hf_hub_id='timm/',
        min_input_size=(3, 128, 128), input_size=(3, 256, 256), pool_size=(8, 8)),
    'lambda_resnet26rpt_256.c1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/lambda_resnet26rpt_c_256-ab00292d.pth',
        hf_hub_id='timm/',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.94),

    'haloregnetz_b.ra3_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/haloregnetz_c_raa_256-c8ad7616.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        first_conv='stem.conv', input_size=(3, 224, 224), pool_size=(7, 7), min_input_size=(3, 224, 224), crop_pct=0.94),

    'lamhalobotnet50ts_256.a1h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/lamhalobotnet50ts_a1h2_256-fe3d9445.pth',
        hf_hub_id='timm/',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'halo2botnet50ts_256.a1h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/halo2botnet50ts_a1h2_256-fd9c11a3.pth',
        hf_hub_id='timm/',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
})


@register_model
def botnet26t_256(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Bottleneck Transformer w/ ResNet26-T backbone.
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('botnet26t_256', 'botnet26t', pretrained=pretrained, **kwargs)


@register_model
def sebotnet33ts_256(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Bottleneck Transformer w/ a ResNet33-t backbone, SE attn for non Halo blocks, SiLU,
    """
    return _create_byoanet('sebotnet33ts_256', 'sebotnet33ts', pretrained=pretrained, **kwargs)


@register_model
def botnet50ts_256(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Bottleneck Transformer w/ ResNet50-T backbone, silu act.
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('botnet50ts_256', 'botnet50ts', pretrained=pretrained, **kwargs)


@register_model
def eca_botnext26ts_256(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Bottleneck Transformer w/ ResNet26-T backbone, silu act.
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('eca_botnext26ts_256', 'eca_botnext26ts', pretrained=pretrained, **kwargs)


@register_model
def halonet_h1(pretrained: bool = False, **kwargs) -> ByobNet:
    """ HaloNet-H1. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h1', pretrained=pretrained, **kwargs)


@register_model
def halonet26t(pretrained: bool = False, **kwargs) -> ByobNet:
    """ HaloNet w/ a ResNet26-t backbone. Halo attention in final two stages
    """
    return _create_byoanet('halonet26t', pretrained=pretrained, **kwargs)


@register_model
def sehalonet33ts(pretrained: bool = False, **kwargs) -> ByobNet:
    """ HaloNet w/ a ResNet33-t backbone, SE attn for non Halo blocks, SiLU, 1-2 Halo in stage 2,3,4.
    """
    return _create_byoanet('sehalonet33ts', pretrained=pretrained, **kwargs)


@register_model
def halonet50ts(pretrained: bool = False, **kwargs) -> ByobNet:
    """ HaloNet w/ a ResNet50-t backbone, silu act. Halo attention in final two stages
    """
    return _create_byoanet('halonet50ts', pretrained=pretrained, **kwargs)


@register_model
def eca_halonext26ts(pretrained: bool = False, **kwargs) -> ByobNet:
    """ HaloNet w/ a ResNet26-t backbone, silu act. Halo attention in final two stages
    """
    return _create_byoanet('eca_halonext26ts', pretrained=pretrained, **kwargs)


@register_model
def lambda_resnet26t(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Lambda-ResNet-26-T. Lambda layers w/ conv pos in last two stages.
    """
    return _create_byoanet('lambda_resnet26t', pretrained=pretrained, **kwargs)


@register_model
def lambda_resnet50ts(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Lambda-ResNet-50-TS. SiLU act. Lambda layers w/ conv pos in last two stages.
    """
    return _create_byoanet('lambda_resnet50ts', pretrained=pretrained, **kwargs)


@register_model
def lambda_resnet26rpt_256(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Lambda-ResNet-26-R-T. Lambda layers w/ rel pos embed in last two stages.
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('lambda_resnet26rpt_256', pretrained=pretrained, **kwargs)


@register_model
def haloregnetz_b(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Halo + RegNetZ
    """
    return _create_byoanet('haloregnetz_b', pretrained=pretrained, **kwargs)


@register_model
def lamhalobotnet50ts_256(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Combo Attention (Lambda + Halo + Bot) Network
    """
    return _create_byoanet('lamhalobotnet50ts_256', 'lamhalobotnet50ts', pretrained=pretrained, **kwargs)


@register_model
def halo2botnet50ts_256(pretrained: bool = False, **kwargs) -> ByobNet:
    """ Combo Attention (Halo + Halo + Bot) Network
    """
    return _create_byoanet('halo2botnet50ts_256', 'halo2botnet50ts', pretrained=pretrained, **kwargs)
