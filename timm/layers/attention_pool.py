from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import maybe_add_mask
from .config import use_fused_attn
from .mlp import Mlp
from .weight_init import trunc_normal_tf_


class AttentionPoolLatent(nn.Module):
    """ Attention pooling w/ latent query

    Setting out_features=0 disables the output projection, norm, and MLP layers (pre_logits mode).
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            feat_size: Optional[int] = None,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = 'token',
            norm_layer: Optional[Type[nn.Module]] = None,
            act_layer: Optional[Type[nn.Module]] = nn.GELU,
            drop: float = 0.0,
            device = None,
            dtype = None
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        embed_dim = embed_dim or in_features
        if out_features is None:
            out_features = in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()

        if pos_embed == 'abs':
            assert feat_size is not None
            self.pos_embed = nn.Parameter(torch.zeros(feat_size, in_features, **dd))
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim, **dd))

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, **dd)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias, **dd)
        if qk_norm:
            qk_norm_layer = norm_layer or nn.LayerNorm
            self.q_norm = qk_norm_layer(self.head_dim, **dd)
            self.k_norm = qk_norm_layer(self.head_dim, **dd)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if out_features > 0:
            self.proj = nn.Linear(embed_dim, out_features, **dd)
            self.proj_drop = nn.Dropout(drop)
            self.norm = norm_layer(out_features, **dd) if norm_layer is not None else nn.Identity()
            self.mlp = Mlp(out_features, int(out_features * mlp_ratio), out_features=out_features, act_layer=act_layer, **dd)
        else:
            self.proj = nn.Identity()
            self.proj_drop = nn.Dropout(drop)
            self.norm = nn.Identity()
            self.mlp = None
            out_features = embed_dim

        self.out_features = out_features

        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.mlp is not None:
            x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x


class AttentionPoolPrr(nn.Module):
    """ Patch Representation Refinement (PRR) attention pool.

    From "Locality-Attending Vision Transformer" (ICLR 2026).

    Parameter-free multi-head self-attention that refines all patch representations
    before pooling. No Q/K/V projections — input is reshaped directly into multi-head
    format for self-attention.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            pool_type: str = 'token',
            pre_norm: bool = False,
            post_norm: bool = False,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        assert pool_type in ('token', 'avg'), f"pool_type must be 'token' or 'avg', got '{pool_type}'"
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        if norm_layer is None and (pre_norm or post_norm):
            norm_layer = nn.LayerNorm

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()
        self.out_features = dim

        self.pre_norm = norm_layer(dim, **dd) if pre_norm else nn.Identity()
        self.post_norm = norm_layer(dim, **dd) if post_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        x = self.pre_norm(x)

        # Parameter-free self-attention: reshape into multi-head format
        qkv = x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(qkv, qkv, qkv)
        else:
            attn = (qkv * self.scale) @ qkv.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ qkv
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.post_norm(x)

        # Pool
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)

        return x