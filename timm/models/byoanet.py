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
import math
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Tuple, List, Optional, Union, Any, Callable
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .byobnet import BlocksCfg, ByobCfg, create_byob_stem, create_byob_stages, create_downsample,\
    reduce_feat_size, register_block, num_groups, LayerFn, _init_weights
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, ConvBnAct, DropPath, get_act_layer, convert_norm_act, get_attn, get_self_attn,\
    make_divisible, to_2tuple
from .registry import register_model

__all__ = ['ByoaNet']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.conv', 'classifier': 'head.fc',
        'fixed_input_size': False, 'min_input_size': (3, 224, 224),
        **kwargs
    }


default_cfgs = {
    # GPU-Efficient (ResNet) weights
    'botnet50t_224': _cfg(url='', fixed_input_size=True),
    'botnet50t_c4c5_224': _cfg(url='', fixed_input_size=True),

    'halonet_h1': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'halonet_h1_c4c5': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'halonet26t': _cfg(url=''),
    'halonet50t': _cfg(url=''),

    'lambda_resnet26t': _cfg(url='', min_input_size=(3, 128, 128)),
    'lambda_resnet50t': _cfg(url='', min_input_size=(3, 128, 128)),
}


@dataclass
class ByoaBlocksCfg(BlocksCfg):
    # FIXME allow overriding self_attn layer or args per block/stage,
    pass


@dataclass
class ByoaCfg(ByobCfg):
    blocks: Tuple[Union[ByoaBlocksCfg, Tuple[ByoaBlocksCfg, ...]], ...] = None
    self_attn_layer: Optional[str] = None
    self_attn_fixed_size: bool = False
    self_attn_kwargs: dict = field(default_factory=lambda: dict())


def interleave_attn(
        types : Tuple[str, str], every: Union[int, List[int]], d, first: bool = False, **kwargs
) -> Tuple[ByoaBlocksCfg]:
    """ interleave attn blocks
    """
    assert len(types) == 2
    if isinstance(every, int):
        every = list(range(0 if first else every, d, every))
        if not every:
            every = [d - 1]
    set(every)
    blocks = []
    for i in range(d):
        block_type = types[1] if i in every else types[0]
        blocks += [ByoaBlocksCfg(type=block_type, d=1, **kwargs)]
    return tuple(blocks)


model_cfgs = dict(

    botnet50t=ByoaCfg(
        blocks=(
            ByoaBlocksCfg(type='bottle', d=3, c=256, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=4, c=512, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=6, c=1024, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='self_attn', d=3, c=2048, s=1, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        num_features=0,
        self_attn_layer='bottleneck',
        self_attn_fixed_size=True,
        self_attn_kwargs=dict()
    ),
    botnet50t_c4c5=ByoaCfg(
        blocks=(
            ByoaBlocksCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=4, c=512, s=2, gs=0, br=0.25),
            (
                ByoaBlocksCfg(type='self_attn', d=1, c=1024, s=2, gs=0, br=0.25),
                ByoaBlocksCfg(type='bottle', d=5, c=1024, s=1, gs=0, br=0.25),
            ),
            (
                ByoaBlocksCfg(type='self_attn', d=1, c=2048, s=2, gs=0, br=0.25),
                ByoaBlocksCfg(type='bottle', d=2, c=2048, s=1, gs=0, br=0.25),
            )
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='bottleneck',
        self_attn_fixed_size=True,
        self_attn_kwargs=dict()
    ),

    halonet_h1=ByoaCfg(
        blocks=(
            ByoaBlocksCfg(type='self_attn', d=3, c=64, s=1, gs=0, br=1.0),
            ByoaBlocksCfg(type='self_attn', d=3, c=128, s=2, gs=0, br=1.0),
            ByoaBlocksCfg(type='self_attn', d=10, c=256, s=2, gs=0, br=1.0),
            ByoaBlocksCfg(type='self_attn', d=3, c=512, s=2, gs=0, br=1.0),
        ),
        stem_chs=64,
        stem_type='7x7',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3),
    ),
    halonet_h1_c4c5=ByoaCfg(
        blocks=(
            ByoaBlocksCfg(type='bottle', d=3, c=64, s=1, gs=0, br=1.0),
            ByoaBlocksCfg(type='bottle', d=3, c=128, s=2, gs=0, br=1.0),
            ByoaBlocksCfg(type='self_attn', d=10, c=256, s=2, gs=0, br=1.0),
            ByoaBlocksCfg(type='self_attn', d=3, c=512, s=2, gs=0, br=1.0),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3),
    ),
    halonet26t=ByoaCfg(
        blocks=(
            ByoaBlocksCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=2, c=1024, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=7, halo_size=2)
    ),
    halonet50t=ByoaCfg(
        blocks=(
            ByoaBlocksCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=4, c=512, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=6, c=1024, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='self_attn', d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=7, halo_size=2)
    ),

    lambda_resnet26t=ByoaCfg(
        blocks=(
            ByoaBlocksCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_attn(types=('bottle', 'self_attn'), every=1, d=2, c=1024, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='lambda',
        self_attn_kwargs=dict()
    ),
    lambda_resnet50t=ByoaCfg(
        blocks=(
            ByoaBlocksCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            ByoaBlocksCfg(type='bottle', d=4, c=512, s=2, gs=0, br=0.25),
            interleave_attn(types=('bottle', 'self_attn'), every=3, d=6, c=1024, s=2, gs=0, br=0.25),
            ByoaBlocksCfg(type='self_attn', d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        num_features=0,
        self_attn_layer='lambda',
        self_attn_kwargs=dict()
    ),
)


@dataclass
class ByoaLayerFn(LayerFn):
    self_attn: Optional[Callable] = None


class SelfAttnBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - optional kxk - self attn - 1x1
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=1., group_size=None,
                 downsample='avg', extra_conv=False, linear_out=False, post_attn_na=True, feat_size=None,
                 layers: ByoaLayerFn = None, drop_block=None, drop_path_rate=0.):
        super(SelfAttnBlock, self).__init__()
        assert layers is not None
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = create_downsample(
                downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation[0],
                apply_act=False, layers=layers)
        else:
            self.shortcut = nn.Identity()

        self.conv1_1x1 = layers.conv_norm_act(in_chs, mid_chs, 1)
        if extra_conv:
            self.conv2_kxk = layers.conv_norm_act(
                mid_chs, mid_chs, kernel_size, stride=stride, dilation=dilation[0],
                groups=groups, drop_block=drop_block)
            stride = 1  # striding done via conv if enabled
        else:
            self.conv2_kxk = nn.Identity()
        opt_kwargs = {} if feat_size is None else dict(feat_size=feat_size)
        # FIXME need to dilate self attn to have dilated network support, moop moop
        self.self_attn = layers.self_attn(mid_chs, stride=stride, **opt_kwargs)
        self.post_attn = layers.norm_act(mid_chs) if post_attn_na else nn.Identity()
        self.conv3_1x1 = layers.conv_norm_act(mid_chs, out_chs, 1, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else layers.act(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        if zero_init_last_bn:
            nn.init.zeros_(self.conv3_1x1.bn.weight)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.self_attn(x)
        x = self.post_attn(x)
        x = self.conv3_1x1(x)
        x = self.drop_path(x)

        x = self.act(x + shortcut)
        return x

register_block('self_attn', SelfAttnBlock)


def _byoa_block_args(block_kwargs, block_cfg: ByoaBlocksCfg, model_cfg: ByoaCfg, feat_size=None):
    if block_cfg.type == 'self_attn' and model_cfg.self_attn_fixed_size:
        assert feat_size is not None
        block_kwargs['feat_size'] = feat_size
    return block_kwargs


def get_layer_fns(cfg: ByoaCfg):
    act = get_act_layer(cfg.act_layer)
    norm_act = convert_norm_act(norm_layer=cfg.norm_layer, act_layer=act)
    conv_norm_act = partial(ConvBnAct, norm_layer=cfg.norm_layer, act_layer=act)
    attn = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None
    self_attn = partial(get_self_attn(cfg.self_attn_layer), **cfg.self_attn_kwargs) if cfg.self_attn_layer else None
    layer_fn = ByoaLayerFn(
        conv_norm_act=conv_norm_act, norm_act=norm_act, act=act, attn=attn, self_attn=self_attn)
    return layer_fn


class ByoaNet(nn.Module):
    """ 'Bring-your-own-attention' Net

    A ResNet inspired backbone that supports interleaving traditional residual blocks with
    'Self Attention' bottleneck blocks that replace the bottleneck kxk conv w/ a self-attention
    or similar module.

    FIXME This class network definition is almost the same as ByobNet, I'd like to merge them but
    torchscript limitations prevent sensible inheritance overrides.
    """
    def __init__(self, cfg: ByoaCfg, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
                 zero_init_last_bn=True, img_size=None, drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        layers = get_layer_fns(cfg)
        feat_size = to_2tuple(img_size) if img_size is not None else None

        self.feature_info = []
        stem_chs = int(round((cfg.stem_chs or cfg.blocks[0].c) * cfg.width_factor))
        self.stem, stem_feat = create_byob_stem(in_chans, stem_chs, cfg.stem_type, cfg.stem_pool, layers=layers)
        self.feature_info.extend(stem_feat[:-1])
        feat_size = reduce_feat_size(feat_size, stride=stem_feat[-1]['reduction'])

        self.stages, stage_feat = create_byob_stages(
            cfg, drop_path_rate, output_stride, stem_feat[-1],
            feat_size=feat_size, layers=layers, extra_args_fn=_byoa_block_args)
        self.feature_info.extend(stage_feat[:-1])

        prev_chs = stage_feat[-1]['num_chs']
        if cfg.num_features:
            self.num_features = int(round(cfg.width_factor * cfg.num_features))
            self.final_conv = layers.conv_norm_act(prev_chs, self.num_features, 1)
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.feature_info += [
            dict(num_chs=self.num_features, reduction=stage_feat[-1]['reduction'], module='final_conv')]

        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

        for n, m in self.named_modules():
            _init_weights(m, n)
        for m in self.modules():
            # call each block's weight init for block-specific overrides to init above
            if hasattr(m, 'init_weights'):
                m.init_weights(zero_init_last_bn=zero_init_last_bn)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_byoanet(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByoaNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)


@register_model
def botnet50t_224(pretrained=False, **kwargs):
    """ Bottleneck Transformer w/ ResNet50-T backbone. Bottleneck attn in final stage.
    """
    kwargs.setdefault('img_size', 224)
    return _create_byoanet('botnet50t_224', 'botnet50t', pretrained=pretrained, **kwargs)


@register_model
def botnet50t_c4c5_224(pretrained=False, **kwargs):
    """ Bottleneck Transformer w/ ResNet50-T backbone. Bottleneck attn in last two stages.
    """
    kwargs.setdefault('img_size', 224)
    return _create_byoanet('botnet50t_c4c5_224', 'botnet50t_c4c5', pretrained=pretrained, **kwargs)


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
def halonet50t(pretrained=False, **kwargs):
    """ HaloNet w/ a ResNet50-t backbone, Hallo attention in final stage
    """
    return _create_byoanet('halonet50t', pretrained=pretrained, **kwargs)


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
