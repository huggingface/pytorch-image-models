""" ViTamin

Paper: Designing Scalable Vison Models in the Vision-Language Era
A family of model weights on Huggingface: https://huggingface.co/collections/jienengchen/vitamin-family-661048126b72debdaca060bf

@inproceedings{chen2024vitamin,
  title={ViTamin: Designing Scalable Vision Models in the Vision-language Era},
  author={Chen, Jieneng and Yu, Qihang and Shen, Xiaohui and Yuille, Alan and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

Based on Apache 2.0 licensed code at https://github.com/ViTamin/ViTamin

Modifications and timm support by Jieneng Chen 2024

Reference:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_hybrid.py
"""

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import create_act_layer, get_norm_layer, get_norm_act_layer, create_conv2d, \
    make_divisible, DropPath, HybridEmbed
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from .vision_transformer import VisionTransformer, checkpoint_filter_fn


@dataclass
class VitConvCfg:
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    act_layer: str = 'gelu' # stem & stage 1234
    norm_layer: str = ''
    norm_eps: float = 1e-5
    down_shortcut: Optional[bool] = True
    mlp: str = 'mlp'


@dataclass
class VitCfg:
    embed_dim: Tuple[Union[int, Tuple[int, ...]], ...] = (96, 192, 384, 768)
    depths: Tuple[Union[int, Tuple[int, ...]], ...] = (2, 3, 5, 2)
    stem_width: int = 64
    conv_cfg: VitConvCfg = field(default_factory=VitConvCfg)
    head_type: str = ""


def _init_conv(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Stem(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            act_layer: str = 'gelu',
            norm_layer: str = 'layernorm2d',
            norm_eps: float = 1e-6,
            bias: bool = True,
    ):
        super().__init__()
        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs

        self.conv1 = create_conv2d(in_chs, out_chs, 3, stride=2, bias=bias)
        self.norm1 = norm_act_layer(out_chs)
        self.conv2 = create_conv2d(out_chs, out_chs, 3, stride=1, bias=bias)

        named_apply(_init_conv, self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class Downsample2d(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            pool_type: str = 'avg2',
            bias: bool = True,
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias) # 1x1 conv
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x


class StridedConv(nn.Module):
    """ downsample 2d as well
    """
    def __init__(
            self,
            kernel_size=3,
            stride=2,
            padding=1,
            in_chans=3,
            embed_dim=768
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer('layernorm2d'), eps=1e-6)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer(in_chans) # affine over C

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        return x


class MbConvLNBlock(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            drop_path: float = 0.,
            kernel_size: int = 3,
            norm_layer: str = 'layernorm2d',
            norm_eps: float = 1e-6,
            act_layer: str = 'gelu',
            expand_ratio: float = 4.0,
    ):
        super(MbConvLNBlock, self).__init__()
        self.stride, self.in_chs, self.out_chs = stride, in_chs, out_chs
        mid_chs = make_divisible(out_chs * expand_ratio)
        prenorm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)

        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs, pool_type='avg', bias=True)
        elif in_chs != out_chs:
            self.shortcut = nn.Conv2d(in_chs, out_chs, 1, bias=True)
        else:
            self.shortcut = nn.Identity()

        self.pre_norm = prenorm_act_layer(in_chs, apply_act=False)
        self.down = nn.Identity()
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv2_kxk = create_conv2d(
            mid_chs, mid_chs, kernel_size, stride=stride, dilation=1, groups=mid_chs, bias=True)
        self.act2 = create_act_layer(act_layer, inplace=True)
        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.pre_norm(x)
        x = self.down(x) # nn.Identity()

        # 1x1 expansion conv & act
        x = self.conv1_1x1(x)
        x = self.act1(x)

        # (strided) depthwise 3x3 conv & act
        x = self.conv2_kxk(x)
        x = self.act2(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut

        return x


class MbConvStages(nn.Module):
    """ MobileConv for stage 1 and stage 2 of ViTamin
    """
    def __init__(
            self,
            cfg: VitCfg,
            img_size: Union[int, Tuple[int, int]] = 224, # place holder
            in_chans: int = 3,
    ):
        super().__init__()
        self.grad_checkpointing = False

        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg.stem_width,
        )

        stages = []
        self.num_stages = len(cfg.embed_dim)
        for s, dim in enumerate(cfg.embed_dim[:2]): # stage
            stage_in_chs = cfg.embed_dim[s-1] if s>0 else cfg.stem_width
            blocks = [
                MbConvLNBlock(
                    in_chs = stage_in_chs if d==0 else dim,
                    out_chs = dim,
                    stride = 2 if d == 0 else 1,
                )
                for d in range(cfg.depths[s])
            ]
            stages += [nn.Sequential(*blocks)]
        self.stages = nn.Sequential(*stages)

        self.pool = StridedConv(
            stride=2,
            in_chans=cfg.embed_dim[1],
            embed_dim=cfg.embed_dim[2]
        )

    def forward(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.pool(x)
        return x


class GeGluMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer = 'gelu',
            norm_layer = None,
            bias = True,
            drop = 0.0,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer(norm_layer or 'layernorm'), eps=1e-6)

        self.norm = norm_layer(in_features)
        self.w0 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = create_act_layer(act_layer)
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x


def _create_vitamin(variant, pretrained=False, embed_cfg=None, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    assert embed_cfg is not None
    backbone = MbConvStages(cfg=embed_cfg, in_chans=kwargs.get('in_chans', 3))
    kwargs['embed_layer'] = partial(HybridEmbed, backbone=backbone, proj=False)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': OPENAI_CLIP_MEAN, 'std': OPENAI_CLIP_STD,
        'first_conv': 'patch_embed.backbone.stem.conv1',
        'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'vitamin_small_224.datacomp1b_clip_ltt': _cfg(
        hf_hub_id='jienengchen/ViTamin-S-LTT', num_classes=384),
    'vitamin_small_224.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-S', num_classes=384),
    'vitamin_base_224.datacomp1b_clip_ltt': _cfg(
        hf_hub_id='jienengchen/ViTamin-B-LTT', num_classes=768),
    'vitamin_base_224.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-B', num_classes=768),
    'vitamin_large_224.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-L-224px', num_classes=768),
    'vitamin_large_256.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-L-256px', num_classes=768,
        input_size=(3, 256, 256), crop_pct=1.0),
    'vitamin_large_336.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-L-336px', num_classes=768,
        input_size=(3, 336, 336), crop_pct=1.0),
    'vitamin_large_384.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-L-384px', num_classes=768,
        input_size=(3, 384, 384), crop_pct=1.0),
    'vitamin_large2_224.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-L2-224px', num_classes=1024),
    'vitamin_large2_256.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-L2-256px', num_classes=1024,
        input_size=(3, 256, 256), crop_pct=1.0),
    'vitamin_large2_336.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-L2-336px', num_classes=1024,
        input_size=(3, 336, 336), crop_pct=1.0),
    'vitamin_large2_384.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-L2-384px', num_classes=1024,
        input_size=(3, 384, 384), crop_pct=1.0),
    'vitamin_xlarge_256.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-XL-256px', num_classes=1152,
        input_size=(3, 256, 256), crop_pct=1.0),
    'vitamin_xlarge_336.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-XL-336px', num_classes=1152,
        input_size=(3, 336, 336), crop_pct=1.0),
    'vitamin_xlarge_384.datacomp1b_clip': _cfg(
        hf_hub_id='jienengchen/ViTamin-XL-384px', num_classes=1152,
        input_size=(3, 384, 384), crop_pct=1.0),
})


@register_model
def vitamin_small_224(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(64, 128, 384),
        depths=(2, 4, 1),
        stem_width=64,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        embed_dim=384, depth=14, num_heads=6, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg
    )
    model = _create_vitamin('vitamin_small_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_base_224(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(128, 256, 768),
        depths=(2, 4, 1),
        stem_width=128,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        embed_dim=768, depth=14, num_heads=12, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg)
    model = _create_vitamin('vitamin_base_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large_224(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg,
    )
    model = _create_vitamin('vitamin_large_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large_256(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=256, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg)
    model = _create_vitamin('vitamin_large_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large_336(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=336, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg
    )
    model = _create_vitamin('vitamin_large_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large_384(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=384, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg)
    model = _create_vitamin('vitamin_large_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large2_224(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg,
    )
    model = _create_vitamin('vitamin_large2_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large2_256(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=256, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg)
    model = _create_vitamin('vitamin_large2_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large2_336(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=336, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg
    )
    model = _create_vitamin('vitamin_large2_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large2_384(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=384, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', embed_cfg=embed_cfg)
    model = _create_vitamin('vitamin_large2_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_xlarge_256(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg=VitCfg(
        embed_dim=(192, 384, 1152),
        depths=(2, 4, 1),
        stem_width=192,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=256, embed_dim=1152, depth=32, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', pos_embed='none', embed_cfg=embed_cfg)
    model = _create_vitamin(
        'vitamin_xlarge_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_xlarge_336(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(192, 384, 1152),
        depths=(2, 4, 1),
        stem_width=192,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=336, embed_dim=1152, depth=32, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', pos_embed='none', embed_cfg=embed_cfg)
    model = _create_vitamin('vitamin_xlarge_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_xlarge_384(pretrained=False, **kwargs) -> VisionTransformer:
    embed_cfg = VitCfg(
        embed_dim=(192, 384, 1152),
        depths=(2, 4, 1),
        stem_width=192,
        conv_cfg=VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    )
    model_args = dict(
        img_size=384, embed_dim=1152, depth=32, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2.,
        class_token=False, global_pool='avg', pos_embed='none', embed_cfg=embed_cfg)
    model = _create_vitamin('vitamin_xlarge_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model