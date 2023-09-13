""" Packed Sequence Vision Transformer (ViT) in PyTorch

Base on ideas in NaViT paper
`Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution` - https://arxiv.org/abs/2307.06304

This is a WIP, TODO:
* significant additions to dataset pipeline (data loading / collation) to support sequences required
* token (patch) dropout needs to be implemented
* wider variety of position embedding options

"""
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, trunc_normal_tf_, \
    resample_patch_embed, resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked, to_2tuple
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq
from ._registry import generate_default_cfgs, register_model
from .vision_transformer import get_init_weights_vit

__all__ = ['VisionTransformerPacked']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)


def extract_patches(
        x,
        patch_size=(16, 16),
        channels_last=False,
        flatten_grid=True,
        pad=False,
):
    B, C, H, W = x.shape
    ph, pw = patch_size
    if pad:
        pad_h = (patch_size[0] - H % patch_size[0]) % patch_size[0]
        pad_w = (patch_size[1] - W % patch_size[1]) % patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))
        H += pad_h
        W += pad_w
    gh, gw = H // ph, W // pw
    if channels_last:
        #x = x.unfold(2, ph, pw).unfold(3, ph, pw).permute(0, 2, 3, 4, 5, 1).reshape(B, -1, ph * pw * C)
        x = x.reshape(B, C, gh, ph, gw, pw).permute(0, 2, 4, 3, 5, 1)  # B, gH, gW, pH, pW,  C
    else:
        #x = x.permute(0, 2, 3, 1).unfold(1, ph, pw).unfold(2, ph, pw).reshape(B, -1, C * ph * pw)
        x = x.reshape(B, C, gh, ph, gw, pw).permute(0, 2, 4, 1, 3, 5)
    if flatten_grid:
        x = x.reshape(B, -1, C * ph * pw)
    else:
        x = x.reshape(B, gh, gw, -1)
    return x


@dataclass
class PackedSequence:
    tokens: List[torch.Tensor] = field(default_factory=list)
    pos_indices: List[torch.Tensor] = field(default_factory=list)
    seq_ids: List[torch.Tensor] = field(default_factory=list)
    seq_lens: List[int] = field(default_factory=list)
    total_len: int = 0
    num_images: int = 0

    def add_image(self, tokens, pos_indices):
        seq_id = self.num_images + 1
        seq_len = len(tokens)
        device = tokens.device
        self.tokens.append(tokens)
        self.pos_indices.append(pos_indices)
        self.seq_ids.append(torch.tensor([seq_id] * seq_len, dtype=torch.int64, device=device))
        self.seq_lens.append(seq_len)
        self.total_len += seq_len
        self.num_images += 1

    def to_tensors(self, max_len, max_packed, return_mask=True):
        assert self.total_len > 0
        assert max_len >= self.total_len
        device = self.tokens[-1].device
        dim = self.tokens[-1].shape[-1]
        pad_len = max_len - self.total_len
        seq_pad = max(0, max_packed - len(self.seq_lens))
        seq_lens = self.seq_lens + [0] * seq_pad if seq_pad else self.seq_lens
        seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        if pad_len:
            tokens = self.tokens + [torch.zeros(pad_len, dim, device=device)]
            pos_indices = self.pos_indices + [torch.zeros((pad_len, 2), dtype=torch.int64, device=device)]
            seq_ids = self.seq_ids + [torch.zeros(pad_len, dtype=torch.int64, device=device)]
        else:
            tokens = self.tokens
            pos_indices = self.pos_indices
            seq_ids = self.seq_ids
        tokens = torch.concat(tokens)
        pos_indices = torch.concat(pos_indices)
        seq_ids = torch.concat(seq_ids)
        if return_mask:
            mask = seq_ids != 0
            return tokens, pos_indices, seq_ids, seq_lens, mask
        return tokens, pos_indices, seq_ids, seq_lens


def pack_images(
        images: List[torch.Tensor],
        patch_size: Tuple[int, int],
        max_grid_size: Tuple[int, int],
        pad_patches: bool = False,
        max_images_per_sequence: int = 4,
):
    max_seq_len = max_grid_size[0] * max_grid_size[1]

    # patchify if needed, generate position indices, apply patch drop, record seq lengths
    img_tokens = []
    img_pos_indices = []
    img_seq_lens = []
    for img in images:
        assert img.ndim == 3
        device = img.device
        patches = extract_patches(img.unsqueeze(0), patch_size, flatten_grid=False, pad=pad_patches).squeeze(0)
        grid_h, grid_w, dim = patches.shape
        seq_len = grid_h * grid_w
        if seq_len > max_seq_len:
            _logger.error('Sequence length of image is too large, skipping.')
            continue
        pos_indices = torch.stack(
            torch.meshgrid((
                torch.arange(grid_h, device=device),
                torch.arange(grid_w, device=device)),
                indexing='ij'),
            dim=-1,
        )
        img_tokens.append(patches.flatten(0, 1))
        img_pos_indices.append(pos_indices.flatten(0, 1))
        img_seq_lens.append(seq_len)
    del images

    # sort by seq length largest -> smallest
    img_seq_lens = torch.tensor(img_seq_lens, dtype=torch.long, device=device)
    seq_sort_indices = torch.argsort(img_seq_lens, descending=True)

    packed_sequences: List[PackedSequence] = []  # image sequences packed together
    next_pos = 0
    max_packed = 0
    for _ in range(len(seq_sort_indices)):
        idx_to_pack = seq_sort_indices[next_pos]
        len_to_pack = img_seq_lens[idx_to_pack]
        sequence = None
        for p in packed_sequences:
            # try over existing
            if p.num_images >= max_images_per_sequence or p.total_len + len_to_pack > max_seq_len:
                # will not fit in this sequence
                continue
            sequence = p
            break

        if sequence is None:
            sequence = PackedSequence()  # start fresh sequence
            packed_sequences.append(sequence)

        img_to_pack = img_tokens[idx_to_pack]
        pos_to_pack = img_pos_indices[idx_to_pack]
        sequence.add_image(img_to_pack, pos_to_pack)
        max_packed = max(sequence.num_images, max_packed)
        next_pos += 1

    tensors = [p.to_tensors(max_len=max_seq_len, max_packed=max_packed) for p in packed_sequences]
    o = [torch.stack(t) for t in zip(*tensors)]
    return tuple(o)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if attn_mask is not None:
            assert attn_mask.ndim == 4
            if attn_mask.shape[1] != self.num_heads:
                attn_mask = attn_mask.expand((-1, self.num_heads, -1, -1))

        if self.fused_attn:
            with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn += attn_mask
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


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask: Optional[torch.Tensor]):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ParallelScalingBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,  # NOTE: not used
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self):
        trunc_normal_tf_(self.in_proj.weight, std=(self.head_dim * self.num_heads) ** -0.5)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
                x_attn = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class AttentionPoolLatent(nn.Module):
    """ Attention pooling w/ latent query
    """
    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            flatten_input: bool = True,
            latent_size: int = 1,
            latent_proj: bool = False,
            latent_dim: int = None,
            pos_embed: str = '',
            proj_type: str = '',
            pool_type: str = '',
            norm_layer: Optional[nn.Module] = None,
            drop: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.flatten_input = flatten_input
        self.pool = pool_type

        if pos_embed == 'abs':
            spatial_len = self.feat_size
            self.pos_embed = nn.Parameter(torch.zeros(spatial_len, in_features))
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        latent_size = latent_size or self.feat_size
        self.latent_len = latent_size
        self.latent = nn.Parameter(torch.zeros(self.latent_len, embed_dim))

        if latent_proj:
            self.q = nn.Linear(in_features, embed_dim, bias=qkv_bias)
        else:
            assert not latent_dim or latent_dim == embed_dim
            self.q = None

        self.kv = nn.Linear(in_features, embed_dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()

        if proj_type == 'linear':
            self.proj = nn.Linear(embed_dim, out_features)
            self.proj_drop = nn.Dropout(drop)
        elif proj_type == 'mlp':
            self.proj = Mlp(
                embed_dim,
                hidden_features=embed_dim * 4,
                out_features=out_features,
                drop=drop)
            self.proj_drop = nn.Identity()
        else:
            assert out_features == embed_dim
            self.proj = None
            self.proj_drop = nn.Dropout(drop)

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        trunc_normal_tf_(self.latent, std=self.latent.shape[1] ** -0.5)
        if self.q is not None:
            trunc_normal_tf_(self.q.weight, std=self.q.weight.shape[1] ** -0.5)
            if self.q.bias is not None:
                nn.init.zeros_(self.q.bias)
        trunc_normal_tf_(self.kv.weight, std=self.kv.weight.shape[1] ** -0.5)
        if self.kv.bias is not None:
            nn.init.zeros_(self.kv.bias)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, N, _ = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q = self.latent if self.q is None else self.q(self.latent)
        q = q.reshape(1, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if attn_mask.shape[2] != q.shape[2]:
            # expand latent q to match attention mask, TODO make this less implicit?
            if q.shape[2] == 1:
                q = q.expand(B, -1, attn_mask.shape[2], -1)
            else:
                assert attn_mask.shape[2] % q.shape[2] == 0
                q = q.repeat(1, 1, attn_mask.shape[2] // q.shape[2], 1)
                q = q.expand(B, -1, -1, -1)
        else:
            q = q.expand(B, -1, -1, -1)
        latent_len = q.shape[2]
        x = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = x.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if False:
            with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn += attn_mask
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, latent_len, -1)

        x = self.norm(x)
        if self.proj is not None:
            shortcut = x
            x = self.proj(x)
            x = self.proj_drop(x)
            x = x + shortcut
        else:
            x = self.proj_drop(x)
        if self.pool == 'token':
            x = x[:, 0]
        return x


class VisionTransformerPacked(nn.Module):
    """ Vision Transformer
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'attn')
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.grad_checkpointing = False
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_size = patch_h, patch_w = to_2tuple(patch_size)
        self.img_size = img_h, img_w = to_2tuple(img_size)  # NOTE this === 'maximum size'
        self.grid_size = grid_h, grid_w = img_h // patch_h, img_w // patch_w
        self.max_seq = grid_h * grid_w
        patch_dim_in = in_chans * patch_h * patch_w

        self.patch_embed = nn.Linear(patch_dim_in, embed_dim)
        self.pos_embed_h = nn.Parameter(torch.randn(grid_h, embed_dim) * .02)
        self.pos_embed_w = nn.Parameter(torch.randn(grid_w, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        if global_pool == 'avg':
            self.attn_pool = None
        else:
            # FIXME attention pooling appears less stable in initial trials
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                self.embed_dim,
                num_heads=num_heads,
                pos_embed='',
                latent_proj=True,
                proj_type='',
                norm_layer=norm_layer,
            )

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed_h, std=.02)
        trunc_normal_(self.pos_embed_w, std=.02)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'embeds.pos_embed', 'embeds.cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^embeds',  # stem and embed
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

    def forward_features(
            self,
            tokens: Union[List[torch.Tensor], torch.Tensor],
            pos_indices: Optional[torch.Tensor] = None,
            seq_ids: Optional[torch.Tensor] = None,
            seq_lens: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        if tokens.ndim == 4:
            # B, C, H, W batch tensor will be converted to list and packed
            # for compatibility with common image model usage (and initial testing)
            tokens = tokens.unbind(0)

        if isinstance(tokens, (list, tuple)):
            tokens, pos_indices, seq_ids, seq_lens, padding_mask = pack_images(
                tokens,
                self.patch_size,
                max_grid_size=self.grid_size,
                pad_patches=True,
                max_images_per_sequence=4,
            )

        assert tokens.ndim == 3
        assert pos_indices is not None
        assert seq_ids is not None
        assert seq_lens is not None

        tokens = self.patch_embed(tokens)
        pos_index_h, pos_index_w = pos_indices.unbind(-1)
        pos = self.pos_embed_h[pos_index_h] + self.pos_embed_w[pos_index_w]
        tokens += pos
        tokens = self.pos_drop(tokens)
        tokens = self.norm_pre(tokens)

        if attn_mask is None:
            attn_mask = seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)
            key_padding_mask = (seq_ids != 0).unsqueeze(1)
            attn_mask = attn_mask & key_padding_mask
            attn_mask = attn_mask.unsqueeze(1)

        if attn_mask.dtype == torch.bool:
            dtype = tokens.dtype
            min_val = torch.finfo(dtype).min
            attn_mask = torch.zeros_like(attn_mask, dtype=dtype).masked_fill_(~attn_mask, min_val)

        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     tokens = checkpoint_seq(self.blocks, tokens)
        # else:
        for b in self.blocks:
            tokens = b(tokens, attn_mask=attn_mask)
        tokens = self.norm(tokens)

        device = tokens.device
        max_packing = seq_lens.shape[1]
        seq_id_range = torch.arange(1, 1 + max_packing, device=device)
        unpack_mask = seq_ids.unsqueeze(1) == seq_id_range[:, None]
        seq_lens = seq_lens.reshape(-1)
        valid_rows = seq_lens > 0
        if self.attn_pool is not None:
            unpack_mask = unpack_mask & key_padding_mask
            unpack_mask = unpack_mask.unsqueeze(1)
            unpack_mask = torch.zeros_like(unpack_mask, dtype=tokens.dtype).masked_fill_(
                ~unpack_mask, torch.finfo(tokens.dtype).min)
            tokens = self.attn_pool(tokens, attn_mask=unpack_mask)
            tokens = tokens.reshape(-1, self.embed_dim)
            tokens = tokens[valid_rows]
        else:
            tokens = tokens.unsqueeze(1).expand(-1, max_packing, -1, -1)[unpack_mask]
            tokens = tokens.tensor_split(seq_lens.reshape(-1).cumsum(0)[:sum(valid_rows) - 1].cpu())
            # tokens = tokens.unsqueeze(1) * unpack_mask.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
            # tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
            # seq_lens = seq_lens[valid_rows]
            # tokens = tokens[valid_rows]

        # FIXME sort out this mess, the boundary of features vs head is a bit messy with
        # variable length sequence averaging vs attention pooling...
        return tokens  #, seq_lens

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            if isinstance(x, (list, tuple)):
                x = torch.stack([t.mean(dim=0) for t in x], 0)
            else:
                x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(
            self,
            tokens: Union[List[torch.Tensor], torch.Tensor],
            pos_indices: Optional[torch.Tensor] = None,
            seq_ids: Optional[torch.Tensor] = None,
            seq_lens: Optional[torch.Tensor] = None,
    ):
        x = self.forward_features(
            tokens,
            pos_indices=pos_indices,
            seq_ids=seq_ids,
            seq_lens=seq_lens,
        )
        x = self.forward_head(x)
        return x


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
    'navit_base_patch32_224': _cfg(),
    'navit_base_patch32_384': _cfg(),
    'navit_base_patch16_224': _cfg(),
    'navit_base_patch16_384': _cfg(),
})


def _create_vision_transformer_packed(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    return build_model_with_cfg(
        VisionTransformerPacked,
        variant,
        pretrained,
        #pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )


@register_model
def navit_base_patch32_224(pretrained=False, **kwargs) -> VisionTransformerPacked:
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_packed('navit_base_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def navit_base_patch32_384(pretrained=False, **kwargs) -> VisionTransformerPacked:
    model_args = dict(img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_packed('navit_base_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def navit_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformerPacked:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_packed('navit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def navit_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformerPacked:
    model_args = dict(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_packed('navit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def navit_base_patch16_xp_384(pretrained=False, **kwargs) -> VisionTransformerPacked:
    model_args = dict(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        qk_norm=True, pre_norm=True, block_fn=ParallelScalingBlock)
    model = _create_vision_transformer_packed('navit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
