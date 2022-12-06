""" Attention Pool 2D

Implementations of 2D spatial feature pooling using multi-head attention instead of average pool.

Based on idea in CLIP by OpenAI, licensed Apache 2.0
https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Union, Tuple

import torch
import torch.nn as nn

from .helpers import to_2tuple
from .pos_embed import apply_rot_embed, RotaryEmbedding
from .weight_init import trunc_normal_


class RotAttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling w/ rotary (relative) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: While this impl does not require a fixed feature size, performance at differeing resolutions from
    train varies widely and falls off dramatically. I'm not sure if there is a way around this... -RW
    """
    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 4,
            qkv_bias: bool = True,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_embed = RotaryEmbedding(self.head_dim)

        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        x = x.reshape(B, -1, N).permute(0, 2, 1)

        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)

        x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]

        qc, q = q[:, :, :1], q[:, :, 1:]
        sin_emb, cos_emb = self.pos_embed.get_embed((H, W))
        q = apply_rot_embed(q, sin_emb, cos_emb)
        q = torch.cat([qc, q], dim=2)

        kc, k = k[:, :, :1], k[:, :, 1:]
        k = apply_rot_embed(k, sin_emb, cos_emb)
        k = torch.cat([kc, k], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x[:, 0]


class AttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """
    def __init__(
            self,
            in_features: int,
            feat_size: Union[int, Tuple[int, int]],
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 4,
            qkv_bias: bool = True,
    ):
        super().__init__()

        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.feat_size = to_2tuple(feat_size)
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        spatial_dim = self.feat_size[0] * self.feat_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(spatial_dim + 1, in_features))
        trunc_normal_(self.pos_embed, std=in_features ** -0.5)
        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        assert self.feat_size[0] == H
        assert self.feat_size[1] == W
        x = x.reshape(B, -1, N).permute(0, 2, 1)
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
        x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x[:, 0]
