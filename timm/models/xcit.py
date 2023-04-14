""" Cross-Covariance Image Transformer (XCiT) in PyTorch

Paper:
    - https://arxiv.org/abs/2106.09681

Same as the official implementation, with some minor adaptations, original copyright below
    - https://github.com/facebookresearch/xcit/blob/master/xcit.py

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import math
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, to_2tuple
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_module
from ._registry import register_model, generate_default_cfgs, register_model_deprecations
from .cait import ClassAttn
from .vision_transformer import Mlp

__all__ = ['XCiT']  # model_registry will add each entrypoint fn to this


@register_notrace_module  # reason: FX can't symbolically trace torch.arange in forward method
class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the "Attention is all you Need" paper.
    Based on the official XCiT code
        - https://github.com/facebookresearch/xcit/blob/master/xcit.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B: int, H: int, W: int):
        device = self.token_projection.weight.device
        y_embed = torch.arange(1, H+1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
        x_embed = torch.arange(1, W+1, dtype=torch.float32, device=device).repeat(1, H, 1)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes)
    )


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, act_layer=nn.GELU):
        super().__init__()
        img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                act_layer(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise('For convolutional projection, patch size has to be in [8, 16]')

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, (Hp, Wp)


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows to augment the
    implicit communication performed by the block diagonal scatter attention. Implemented using 2 layers of separable
    3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class ClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            eta=1.,
            tokens_norm=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim))
            self.gamma2 = nn.Parameter(eta * torch.ones(dim))
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        self.tokens_norm = tokens_norm

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Result of next line is (qkv, B, num (H)eads,  (C')hannels per head, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        # Paper section 3.2 l2-Normalization and temperature scaling
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, C', N), permute -> (B, N, H, C')
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class XCABlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            eta=1.,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma3 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        # NOTE official code has 3 then 2, so keeping it the same to be consistent with loaded weights
        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class XCiT(nn.Module):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            act_layer=None,
            norm_layer=None,
            cls_attn_layers=2,
            use_pos_embed=True,
            eta=1.,
            tokens_norm=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate after positional embedding, and in XCA/CA projection + MLP
            pos_drop_rate: position embedding dropout rate
            proj_drop_rate (float): projection dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate (constant across all layers)
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos_embed: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA

        Notes:
            - Although `layer_norm` is user specifiable, there are hard-coded `BatchNorm2d`s in the local patch
              interaction (class LPI) and the patch embedding (class ConvPatchEmbed)
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        img_size = to_2tuple(img_size)
        assert (img_size[0] % patch_size == 0) and (img_size[0] % patch_size == 0), \
            '`patch_size` should divide image dimensions evenly'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.grad_checkpointing = False

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            act_layer=act_layer,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.blocks = nn.ModuleList([
            XCABlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                eta=eta,
            )
            for _ in range(depth)])

        self.cls_attn_blocks = nn.ModuleList([
            ClassAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                eta=eta,
                tokens_norm=tokens_norm,
            )
            for _ in range(cls_attn_layers)])

        # Classifier head
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Init weights
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=r'^blocks\.(\d+)',
            cls_attn_blocks=[(r'^cls_attn_blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # `pos_embed` (B, C, Hp, Wp), reshape -> (B, C, N), permute -> (B, N, C)
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, Hp, Wp)
            else:
                x = blk(x, Hp, Wp)

        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        for blk in self.cls_attn_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    if 'model' in state_dict:
        state_dict = state_dict['model']
    # For consistency with timm's transformer models while being compatible with official weights source we rename
    # pos_embeder to pos_embed. Also account for use_pos_embed == False
    use_pos_embed = getattr(model, 'pos_embed', None) is not None
    pos_embed_keys = [k for k in state_dict if k.startswith('pos_embed')]
    for k in pos_embed_keys:
        if use_pos_embed:
            state_dict[k.replace('pos_embeder.', 'pos_embed.')] = state_dict.pop(k)
        else:
            del state_dict[k]
    # timm's implementation of class attention in CaiT is slightly more efficient as it does not compute query vectors
    # for all tokens, just the class token. To use official weights source we must split qkv into q, k, v
    if 'cls_attn_blocks.0.attn.qkv.weight' in state_dict and 'cls_attn_blocks.0.attn.q.weight' in model.state_dict():
        num_ca_blocks = len(model.cls_attn_blocks)
        for i in range(num_ca_blocks):
            qkv_weight = state_dict.pop(f'cls_attn_blocks.{i}.attn.qkv.weight')
            qkv_weight = qkv_weight.reshape(3, -1, qkv_weight.shape[-1])
            for j, subscript in enumerate('qkv'):
                state_dict[f'cls_attn_blocks.{i}.attn.{subscript}.weight'] = qkv_weight[j]
            qkv_bias = state_dict.pop(f'cls_attn_blocks.{i}.attn.qkv.bias', None)
            if qkv_bias is not None:
                qkv_bias = qkv_bias.reshape(3, -1)
                for j, subscript in enumerate('qkv'):
                    state_dict[f'cls_attn_blocks.{i}.attn.{subscript}.bias'] = qkv_bias[j]
    return state_dict


def _create_xcit(variant, pretrained=False, default_cfg=None, **kwargs):
    model = build_model_with_cfg(
        XCiT,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj.0.0', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # Patch size 16
    'xcit_nano_12_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p16_224.pth'),
    'xcit_nano_12_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p16_224_dist.pth'),
    'xcit_nano_12_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_tiny_12_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p16_224.pth'),
    'xcit_tiny_12_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p16_224_dist.pth'),
    'xcit_tiny_12_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_tiny_24_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p16_224.pth'),
    'xcit_tiny_24_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p16_224_dist.pth'),
    'xcit_tiny_24_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_small_12_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_224.pth'),
    'xcit_small_12_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_224_dist.pth'),
    'xcit_small_12_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_small_24_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p16_224.pth'),
    'xcit_small_24_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p16_224_dist.pth'),
    'xcit_small_24_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_medium_24_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p16_224.pth'),
    'xcit_medium_24_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p16_224_dist.pth'),
    'xcit_medium_24_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p16_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_large_24_p16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p16_224.pth'),
    'xcit_large_24_p16_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p16_224_dist.pth'),
    'xcit_large_24_p16_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p16_384_dist.pth', input_size=(3, 384, 384)),

    # Patch size 8
    'xcit_nano_12_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p8_224.pth'),
    'xcit_nano_12_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p8_224_dist.pth'),
    'xcit_nano_12_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_nano_12_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_tiny_12_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p8_224.pth'),
    'xcit_tiny_12_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p8_224_dist.pth'),
    'xcit_tiny_12_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_12_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_tiny_24_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p8_224.pth'),
    'xcit_tiny_24_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p8_224_dist.pth'),
    'xcit_tiny_24_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_small_12_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p8_224.pth'),
    'xcit_small_12_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p8_224_dist.pth'),
    'xcit_small_12_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_small_24_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p8_224.pth'),
    'xcit_small_24_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p8_224_dist.pth'),
    'xcit_small_24_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_medium_24_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p8_224.pth'),
    'xcit_medium_24_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p8_224_dist.pth'),
    'xcit_medium_24_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p8_384_dist.pth', input_size=(3, 384, 384)),
    'xcit_large_24_p8_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p8_224.pth'),
    'xcit_large_24_p8_224.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p8_224_dist.pth'),
    'xcit_large_24_p8_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p8_384_dist.pth', input_size=(3, 384, 384)),
})


@register_model
def xcit_nano_12_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False)
    model = _create_xcit('xcit_nano_12_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_nano_12_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False, img_size=384)
    model = _create_xcit('xcit_nano_12_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_12_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_tiny_12_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_12_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_tiny_12_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_12_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_12_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_24_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=24, num_heads=4, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_tiny_24_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_24_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=24, num_heads=4, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_tiny_24_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_24_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_small_24_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_24_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_small_24_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_medium_24_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=512, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_medium_24_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_medium_24_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=512, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_medium_24_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_large_24_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=24, num_heads=16, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_large_24_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_large_24_p16_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=24, num_heads=16, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_large_24_p16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# Patch size 8x8 models
@register_model
def xcit_nano_12_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False)
    model = _create_xcit('xcit_nano_12_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_nano_12_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=128, depth=12, num_heads=4, eta=1.0, tokens_norm=False)
    model = _create_xcit('xcit_nano_12_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_12_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=192, depth=12, num_heads=4, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_tiny_12_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_12_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=192, depth=12, num_heads=4, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_tiny_12_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_12_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_12_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_24_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=192, depth=24, num_heads=4, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_tiny_24_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_tiny_24_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=192, depth=24, num_heads=4, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_tiny_24_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_24_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=384, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_small_24_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_small_24_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=384, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_small_24_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_medium_24_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=512, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_medium_24_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_medium_24_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=512, depth=24, num_heads=8, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_medium_24_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_large_24_p8_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=768, depth=24, num_heads=16, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_large_24_p8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def xcit_large_24_p8_384(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=8, embed_dim=768, depth=24, num_heads=16, eta=1e-5, tokens_norm=True)
    model = _create_xcit('xcit_large_24_p8_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


register_model_deprecations(__name__, {
    # Patch size 16
    'xcit_nano_12_p16_224_dist': 'xcit_nano_12_p16_224.fb_dist_in1k',
    'xcit_nano_12_p16_384_dist': 'xcit_nano_12_p16_384.fb_dist_in1k',
    'xcit_tiny_12_p16_224_dist': 'xcit_tiny_12_p16_224.fb_dist_in1k',
    'xcit_tiny_12_p16_384_dist': 'xcit_tiny_12_p16_384.fb_dist_in1k',
    'xcit_tiny_24_p16_224_dist': 'xcit_tiny_24_p16_224.fb_dist_in1k',
    'xcit_tiny_24_p16_384_dist': 'xcit_tiny_24_p16_384.fb_dist_in1k',
    'xcit_small_12_p16_224_dist': 'xcit_small_12_p16_224.fb_dist_in1k',
    'xcit_small_12_p16_384_dist': 'xcit_small_12_p16_384.fb_dist_in1k',
    'xcit_small_24_p16_224_dist': 'xcit_small_24_p16_224.fb_dist_in1k',
    'xcit_medium_24_p16_224_dist': 'xcit_medium_24_p16_224.fb_dist_in1k',
    'xcit_medium_24_p16_384_dist': 'xcit_medium_24_p16_384.fb_dist_in1k',
    'xcit_large_24_p16_224_dist': 'xcit_large_24_p16_224.fb_dist_in1k',
    'xcit_large_24_p16_384_dist': 'xcit_large_24_p16_384.fb_dist_in1k',

    # Patch size 8
    'xcit_nano_12_p8_224_dist': 'xcit_nano_12_p8_224.fb_dist_in1k',
    'xcit_nano_12_p8_384_dist': 'xcit_nano_12_p8_384.fb_dist_in1k',
    'xcit_tiny_12_p8_224_dist': 'xcit_tiny_12_p8_224.fb_dist_in1k',
    'xcit_tiny_12_p8_384_dist': 'xcit_tiny_12_p8_384.fb_dist_in1k',
    'xcit_tiny_24_p8_224_dist': 'xcit_tiny_24_p8_224.fb_dist_in1k',
    'xcit_tiny_24_p8_384_dist': 'xcit_tiny_24_p8_384.fb_dist_in1k',
    'xcit_small_12_p8_224_dist': 'xcit_small_12_p8_224.fb_dist_in1k',
    'xcit_small_12_p8_384_dist': 'xcit_small_12_p8_384.fb_dist_in1k',
    'xcit_small_24_p8_224_dist': 'xcit_small_24_p8_224.fb_dist_in1k',
    'xcit_small_24_p8_384_dist': 'xcit_small_24_p8_384.fb_dist_in1k',
    'xcit_medium_24_p8_224_dist': 'xcit_medium_24_p8_224.fb_dist_in1k',
    'xcit_medium_24_p8_384_dist': 'xcit_medium_24_p8_384.fb_dist_in1k',
    'xcit_large_24_p8_224_dist': 'xcit_large_24_p8_224.fb_dist_in1k',
    'xcit_large_24_p8_384_dist': 'xcit_large_24_p8_384.fb_dist_in1k',
})
