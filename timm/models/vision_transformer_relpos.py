""" Relative Position Vision Transformer (ViT) in PyTorch

NOTE: these models are experimental / WIP, expect changes

Hacked together by / Copyright 2022, Ross Wightman
"""
import logging
import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.jit import Final
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import PatchEmbed, Mlp, DropPath, RelPosMlp, RelPosBias, use_fused_attn
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model

__all__ = ['VisionTransformerRelPos']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)


class RelPosAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            rel_pos_cls=None,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.rel_pos = rel_pos_cls(num_heads=num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            attn_bias = None
            if self.rel_pos is not None:
                attn_bias = self.rel_pos.get_bias()
            elif shared_rel_pos is not None:
                attn_bias = shared_rel_pos

            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.rel_pos is not None:
                attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
            elif shared_rel_pos is not None:
                attn = attn + shared_rel_pos
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RelPosBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            rel_pos_cls=None,
            init_values=None,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RelPosAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            rel_pos_cls=rel_pos_cls,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostRelPosBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            rel_pos_cls=None,
            init_values=None,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.init_values = init_values

        self.attn = RelPosAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            rel_pos_cls=rel_pos_cls,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class VisionTransformerRelPos(nn.Module):
    """ Vision Transformer w/ Relative Position Bias

    Differing from classic vit, this impl
      * uses relative position index (swin v1 / beit) or relative log coord + mlp (swin v2) pos embed
      * defaults to no class token (can be enabled)
      * defaults to global avg pool for head (can be changed)
      * layer-scale (residual branch gain) enabled
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_norm=False,
            init_values=1e-6,
            class_token=False,
            fc_norm=False,
            rel_pos_type='mlp',
            rel_pos_dim=None,
            shared_rel_pos=False,
            drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='skip',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=RelPosBlock
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'avg')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_norm (bool): Enable normalization of query and key in attention
            init_values: (float): layer-scale init values
            class_token (bool): use class token (default: False)
            fc_norm (bool): use pre classifier norm instead of pre-pool
            rel_pos_ty pe (str): type of relative position
            shared_rel_pos (bool): share relative pos across all blocks
            drop_rate (float): dropout rate
            proj_drop_rate (float): projection dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        feat_size = self.patch_embed.grid_size

        rel_pos_args = dict(window_size=feat_size, prefix_tokens=self.num_prefix_tokens)
        if rel_pos_type.startswith('mlp'):
            if rel_pos_dim:
                rel_pos_args['hidden_dim'] = rel_pos_dim
            if 'swin' in rel_pos_type:
                rel_pos_args['mode'] = 'swin'
            rel_pos_cls = partial(RelPosMlp, **rel_pos_args)
        else:
            rel_pos_cls = partial(RelPosBias, **rel_pos_args)
        self.shared_rel_pos = None
        if shared_rel_pos:
            self.shared_rel_pos = rel_pos_cls(num_heads=num_heads)
            # NOTE shared rel pos currently mutually exclusive w/ per-block, but could support both...
            rel_pos_cls = None

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim)) if class_token else None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                rel_pos_cls=rel_pos_cls,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'moco', '')
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        # FIXME weight init scheme using PyTorch defaults curently
        #named_apply(get_init_weights_vit(mode, head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        shared_rel_pos = self.shared_rel_pos.get_bias() if self.shared_rel_pos is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos=shared_rel_pos)
            else:
                x = blk(x, shared_rel_pos=shared_rel_pos)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_vision_transformer_relpos(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(VisionTransformerRelPos, variant, pretrained, **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'vit_relpos_base_patch32_plus_rpn_256.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_replos_base_patch32_plus_rpn_256-sw-dd486f51.pth',
        hf_hub_id='timm/',
        input_size=(3, 256, 256)),
    'vit_relpos_base_patch16_plus_240.untrained': _cfg(url='', input_size=(3, 240, 240)),

    'vit_relpos_small_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_small_patch16_224-sw-ec2778b4.pth',
        hf_hub_id='timm/'),
    'vit_relpos_medium_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_medium_patch16_224-sw-11c174af.pth',
        hf_hub_id='timm/'),
    'vit_relpos_base_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_base_patch16_224-sw-49049aed.pth',
        hf_hub_id='timm/'),

    'vit_srelpos_small_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_srelpos_small_patch16_224-sw-6cdb8849.pth',
        hf_hub_id='timm/'),
    'vit_srelpos_medium_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_srelpos_medium_patch16_224-sw-ad702b8c.pth',
        hf_hub_id='timm/'),

    'vit_relpos_medium_patch16_cls_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_medium_patch16_cls_224-sw-cfe8e259.pth',
        hf_hub_id='timm/'),
    'vit_relpos_base_patch16_cls_224.untrained': _cfg(),
    'vit_relpos_base_patch16_clsgap_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_base_patch16_gapcls_224-sw-1a341d6c.pth',
        hf_hub_id='timm/'),

    'vit_relpos_small_patch16_rpn_224.untrained': _cfg(),
    'vit_relpos_medium_patch16_rpn_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_medium_patch16_rpn_224-sw-5d2befd8.pth',
        hf_hub_id='timm/'),
    'vit_relpos_base_patch16_rpn_224.untrained': _cfg(),
})


@register_model
def vit_relpos_base_patch32_plus_rpn_256(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32+) w/ relative log-coord position and residual post-norm, no class token
    """
    model_args = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, block_fn=ResPostRelPosBlock)
    model = _create_vision_transformer_relpos(
        'vit_relpos_base_patch32_plus_rpn_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_base_patch16_plus_240(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16+) w/ relative log-coord position, no class token
    """
    model_args = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14)
    model = _create_vision_transformer_relpos(
        'vit_relpos_base_patch16_plus_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position, no class token
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, fc_norm=True)
    model = _create_vision_transformer_relpos(
        'vit_relpos_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_medium_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, qkv_bias=False, fc_norm=True)
    model = _create_vision_transformer_relpos(
        'vit_relpos_medium_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, fc_norm=True)
    model = _create_vision_transformer_relpos(
        'vit_relpos_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_srelpos_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ shared relative log-coord position, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, fc_norm=False,
        rel_pos_dim=384, shared_rel_pos=True)
    model = _create_vision_transformer_relpos(
        'vit_srelpos_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_srelpos_medium_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ shared relative log-coord position, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, qkv_bias=False, fc_norm=False,
        rel_pos_dim=512, shared_rel_pos=True)
    model = _create_vision_transformer_relpos(
        'vit_srelpos_medium_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_medium_patch16_cls_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-M/16) w/ relative log-coord position, class token present
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, qkv_bias=False, fc_norm=False,
        rel_pos_dim=256, class_token=True, global_pool='token')
    model = _create_vision_transformer_relpos(
        'vit_relpos_medium_patch16_cls_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_base_patch16_cls_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position, class token present
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, class_token=True, global_pool='token')
    model = _create_vision_transformer_relpos(
        'vit_relpos_base_patch16_cls_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_base_patch16_clsgap_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position, class token present
    NOTE this config is a bit of a mistake, class token was enabled but global avg-pool w/ fc-norm was not disabled
    Leaving here for comparisons w/ a future re-train as it performs quite well.
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, fc_norm=True, class_token=True)
    model = _create_vision_transformer_relpos(
        'vit_relpos_base_patch16_clsgap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_small_patch16_rpn_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position and residual post-norm, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, block_fn=ResPostRelPosBlock)
    model = _create_vision_transformer_relpos(
        'vit_relpos_small_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_medium_patch16_rpn_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position and residual post-norm, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, qkv_bias=False, block_fn=ResPostRelPosBlock)
    model = _create_vision_transformer_relpos(
        'vit_relpos_medium_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_relpos_base_patch16_rpn_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position and residual post-norm, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, block_fn=ResPostRelPosBlock)
    model = _create_vision_transformer_relpos(
        'vit_relpos_base_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
