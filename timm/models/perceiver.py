""" Perceiver

Paper: `Perceiver: General Perception with Iterative Attention` - https://arxiv.org/abs/2103.03206

Official Deepmind code: TBD (doesn't exist yet)

Fourier feature position embedding references:
 * Official NeRF impl - https://github.com/bmild/nerf
 * Lucidrain's Perceiver impl - https://github.com/lucidrains/perceiver-pytorch

Status:
* Work in progress, currently running training trials with S and M models (rather slow)

Hacked together by / copyright Ross Wightman, 2021.
"""
import math
from collections import OrderedDict
from functools import partial
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg, named_apply
from .layers import Mlp, DropPath, trunc_normal_, lecun_normal_, to_ntuple, ConvBnAct, LayerNorm2d
from .registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': None, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'perceiver_ss': _cfg(
        url='', input_size=(3, 192, 192)),
    'perceiver_s': _cfg(
        url='', input_size=(3, 192, 192)),
    'perceiver_m': _cfg(
        url=''),
    'perceiver_l': _cfg(
        url=''),
}


def fourier_encode(x, max_freq_log2: int = 8, num_bands: int = 64):
    """ Fourier feature embedding.
    Referenced official NeRF code and Lucidrain's PyTorch Perceiver impl.
    """
    # FIXME this will likely need to change once official code / weights are available
    x = x.unsqueeze(-1)
    bands = 2 ** torch.linspace(0, max_freq_log2 - 1, num_bands, device=x.device, dtype=x.dtype)
    x_bands = x * math.pi * bands
    x = torch.cat([x, x_bands.sin(), x_bands.cos()], dim=-1)
    return x


def fourier_grid(
        shape: List[int], max_freq_log2: int = 8, num_bands: int = 64, device: torch.device = torch.device('cuda')):
    grid = torch.stack(torch.meshgrid([torch.linspace(-1., 1., steps=s, device=device) for s in shape]), dim=-1)
    enc_pos = fourier_encode(grid, max_freq_log2, num_bands)
    return enc_pos.transpose(-1, -2).flatten(len(shape))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    """
    def __init__(self, latent_dim, data_dim, attn_dim=None, num_heads=1, qkv_bias=True, proj_drop=0.):
        super().__init__()
        assert latent_dim % num_heads == 0, f"dim {latent_dim} should be divided by num_heads {num_heads}."

        self.latent_dim = latent_dim
        self.attn_dim = attn_dim or min(latent_dim, data_dim)
        self.num_heads = num_heads
        head_dim = self.attn_dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(latent_dim, self.attn_dim, bias=qkv_bias)
        self.kv = nn.Linear(data_dim, self.attn_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(self.attn_dim, latent_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, latent, data):
        B = latent.shape[0]
        q = self.q(latent).reshape(B, -1, self.num_heads, self.attn_dim // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(data).reshape(B, -1, 2, self.num_heads, self.attn_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, -1, self.attn_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


@torch.jit.interface
class CrossInterface(torch.nn.Module):
    def forward(self, latent: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        pass


class CrossBlock(nn.Module):

    def __init__(self, latent_dim, data_dim, num_heads, attn_dim=None, mlp_ratio=4., qkv_bias=True,
                 drop=0., drop_path=0., attn_layer=CrossAttention, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_latent = norm_layer(latent_dim)
        self.norm1_data = norm_layer(data_dim)
        self.attn = attn_layer(
            latent_dim, data_dim, num_heads=num_heads, attn_dim=attn_dim, qkv_bias=qkv_bias, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(latent_dim)
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        self.mlp = Mlp(in_features=latent_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, latent: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        latent = latent + self.drop_path(self.attn(
            self.norm1_latent(latent),
            self.norm1_data(data),
        ))
        latent = latent + self.drop_path(self.mlp(self.norm2(latent)))
        return latent


@torch.jit.interface
class TransformerInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., attn_layer=Attention, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerStack(nn.Module):
    """ A stack-o-transformers
    NOTE this could have been a simple nn.Sequential but needed to wrap in module to use Interface
    def for ModuleDict torchscript compat.
    """
    def __init__(self, depth, dim, num_heads, mlp_ratio=4., **kwargs):
        super().__init__()
        self.stack = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


def get_layer_layout(cross_depths, num_stages=8, share_weights=None):
    if isinstance(cross_depths, (tuple, list)):
        stage_cross_depths = tuple(cross_depths)
        stage_cross_depths = (stage_cross_depths + (0,) * num_stages)[:num_stages]
    else:
        stage_cross_depths = to_ntuple(num_stages)(cross_depths)
    prev_cross_key = ''
    prev_transformer_key = ''
    keys = []
    num_cross = 0
    num_transformer = 0
    for i, cd in enumerate(stage_cross_depths):
        for j in range(cd):
            key = prev_cross_key
            if share_weights is None or num_cross <= share_weights[0]:
                key = f'c{i}_{j}'
            keys += [key]
            prev_cross_key = key
            num_cross += 1
        key = prev_transformer_key
        if share_weights is None or num_transformer <= share_weights[1]:
            key = f't{i}'
        keys += [key]
        prev_transformer_key = key
        num_transformer += 1
    return keys


class Perceiver(nn.Module):
    """ Perceiver

    Paper: `Perceiver: General Perception with Iterative Attention` - https://arxiv.org/abs/2103.03206
    """

    def __init__(
            self, in_chans=3, num_classes=1000, num_stages=8, cross_depths=(1,), transformer_depth=6,
            latent_dim=1024, num_latents=512, num_latent_heads=8, latent_mlp_ratio=1.0,
            cross_attn_dim=None, num_cross_heads=1, cross_mlp_ratio=1.0, share_weights=(1, 0),
            pos_embed_type='fourier', pos_embed_dim=128, data_bands=64, data_ndim=2, data_max_freq=10,
            data_spatial=False, qkv_bias=True, cross_attn_layer=None, attn_layer=None, norm_layer=None, act_layer=None,
            drop_rate=0., drop_path_rate=0., weight_init=''):
        """
        Args:
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            num_stages (int): number of stages (cross + transformer stack repeats)
            num_cross_heads (int): number of cross-attention heads
            cross_mlp_ratio (flaot): ratio of mlp hidden dim to embedding dim
            share_weights (Optiona[Tuple]): starting index of latent and transformer share (or None for no share)
            latent_dim (int):
            num_latents (int): 
            num_latent_heads (int): number of latent-attention heads
            latent_mlp_ratio (float):
            qkv_bias (bool): enable bias for qkv if True
            pos_embed_type (str): type of pos embed (TODO: currently defaults to fourier)
            pos_embed_dim (int): embedding dimension (for other pos-embed options besides fourier)
            data_bands (int):
            data_ndim (int):
            data_max_freq (int):
            drop_rate (float): dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.latent_dim = latent_dim
        cross_attn_layer = cross_attn_layer or CrossAttention
        attn_layer = attn_layer or Attention
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.latents = nn.Parameter(torch.zeros(num_latents, latent_dim))
        self.data_bands = data_bands
        self.data_max_freq = data_max_freq
        self.data_ndim = data_ndim
        self.data_dim = self.data_ndim * (2 * self.data_bands + 1) + in_chans
        self.data_spatial = data_spatial

        self.blocks_cross = nn.ModuleDict()
        self.blocks_trans = nn.ModuleDict()
        self.layer_keys = get_layer_layout(cross_depths, num_stages, share_weights)
        for i, k in enumerate(self.layer_keys):
            stage_args = dict(
                qkv_bias=qkv_bias, drop=drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer)
            if k.startswith('c'):
                self.blocks_cross[k] = CrossBlock(
                    latent_dim=latent_dim, data_dim=self.data_dim, attn_dim=cross_attn_dim, num_heads=num_cross_heads,
                    mlp_ratio=cross_mlp_ratio, attn_layer=cross_attn_layer, **stage_args)
            else:
                self.blocks_trans[k] = TransformerStack(
                    depth=transformer_depth, dim=latent_dim, num_heads=num_latent_heads,
                    mlp_ratio=latent_mlp_ratio, attn_layer=attn_layer, **stage_args)

        self.norm = norm_layer(latent_dim)
        self.head = nn.Linear(latent_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.latents, std=.02)
        named_apply(partial(_init_weights, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'latents'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.latent_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, C, H, W = x.shape
        # FIXME cache fourier embedding and implement positional options
        # FIXME support ndim inputs, don't assume 2D?
        data = fourier_grid(x.shape[2:], max_freq_log2=self.data_max_freq, num_bands=self.data_bands, device=x.device)
        if self.data_spatial:
            data = torch.cat([x, data.unsqueeze(0).expand(B, -1, -1, -1).permute(0, 3, 1, 2)], dim=1)
        else:
            data = torch.cat([x.permute(0, 2, 3, 1), data.unsqueeze(0).expand(B, -1, -1, -1)], dim=-1)
            data = data.reshape(B, H * W, -1)
        x = self.latents.unsqueeze(0).expand(B, -1, -1)
        for k in self.layer_keys:
            if k.startswith('c'):
                cross_blocks: CrossInterface = self.blocks_cross[k]  # interface annotation for torchscript sillyness
                x = cross_blocks.forward(x, data)
            else:
                transformer: TransformerInterface = self.blocks_trans[k]
                x = transformer.forward(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ weight initialization
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def _create_perceiver(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model = build_model_with_cfg(
        Perceiver, variant, pretrained,
        default_cfg=default_cfg,
        **kwargs)
    return model


@register_model
def perceiver_ss(pretrained=False, **kwargs):
    """ Perceiver-Small (Shared)
     One initial cross attn and all transformer stacks shared. ~11M params
     """
    model_kwargs = dict(
        cross_depths=(1,), latent_dim=512, num_latents=256, cross_attn_dim=128, data_bands=36, **kwargs)
    model = _create_perceiver('perceiver_ss', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def perceiver_s(pretrained=False, **kwargs):
    """ Perceiver-Small
    One initial cross attn and all but first transformer stacks shared. ~20M params
    """
    model_kwargs = dict(
        cross_depths=(1,), latent_dim=512, num_latents=256, cross_attn_dim=128, data_bands=36,
        share_weights=(1, 1), **kwargs)
    model = _create_perceiver('perceiver_s', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def perceiver_m(pretrained=False, **kwargs):
    """ Perceiver-Medium
     Two cross attn (one per each initial transformer stack), all transformers shared. ~25M params.
     """
    model_kwargs = dict(
        cross_depths=(1,) * 2, latent_dim=768, num_latents=384, cross_attn_dim=160, data_bands=40, **kwargs)
    model = _create_perceiver('perceiver_m', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def perceiver_l(pretrained=False, **kwargs):
    """ Perceiver-Large
    One cross attn per 8 transformer stacks. All but first cross attn shared, all transformer stacks shared.
    This variant is closest to the paper model for reported ImageNet results. ~45M params.
    """
    model_kwargs = dict(cross_depths=1, latent_dim=1024, num_latents=512, **kwargs)
    model = _create_perceiver('perceiver_l', pretrained=pretrained, **model_kwargs)
    return model