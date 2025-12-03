""" Non-Local Attention Pooling Layers

A collection of global pooling layers that go beyond simple avg/max pooling.

LSEPool - LogSumExp pooling, a smooth approximation between avg and max pooling
SimPool - Attention-based pooling from 'Keep It SimPool' (ICCV 2023)

Based on implementations from:
* LSE Pooling: custom implementation by Bill Psomas
* SimPool: https://arxiv.org/abs/2309.06891 - 'Keep It SimPool: Who Said Supervised Transformers
    Suffer from Attention Deficit?' by Bill Psomas et al.

Hacked together by / Copyright 2024 Ross Wightman, original code by Bill Psomas
"""
from typing import Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import use_fused_attn


class LsePlus2d(nn.Module):
    """LogSumExp (LSE) Pooling for 2D inputs.

    A smooth approximation to max pooling that provides a learnable interpolation between
    average and max pooling. When r is large, LSE approaches max pooling; when r is small,
    it approaches average pooling.

    Implements: (1/r) * log((1/n) * sum(exp(r * (x - x_max)))) + x_max

    The x_max subtraction provides numerical stability.
    """

    def __init__(
            self,
            r: float = 10.0,
            r_learnable: bool = True,
            flatten: bool = False,
            device=None,
            dtype=None,
    ):
        """
        Args:
            r: Initial value of the pooling parameter. Higher = closer to max pooling.
            r_learnable: If True, r is a learnable parameter.
            flatten: If True, flatten spatial dims in output.
        """
        super().__init__()
        if r_learnable:
            self.r = nn.Parameter(torch.tensor(r, device=device, dtype=dtype))
        else:
            self.register_buffer('r', torch.tensor(r, device=device, dtype=dtype))
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = F.adaptive_max_pool2d(x, 1)
        exp_x = torch.exp(self.r * (x - x_max))
        sum_exp = exp_x.mean(dim=(2, 3), keepdim=True)
        out = x_max + (1.0 / self.r) * torch.log(sum_exp)
        if self.flatten:
            out = out.flatten(1)
        return out


class LsePlus1d(nn.Module):
    """LogSumExp (LSE) Pooling for sequence (NLC) inputs.

    A smooth approximation to max pooling that provides a learnable interpolation between
    average and max pooling. When r is large, LSE approaches max pooling; when r is small,
    it approaches average pooling.
    """

    def __init__(
            self,
            r: float = 10.0,
            r_learnable: bool = True,
            device=None,
            dtype=None,
    ):
        """
        Args:
            r: Initial value of the pooling parameter. Higher = closer to max pooling.
            r_learnable: If True, r is a learnable parameter.
        """
        super().__init__()
        if r_learnable:
            self.r = nn.Parameter(torch.tensor(r, device=device, dtype=dtype))
        else:
            self.register_buffer('r', torch.tensor(r, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x_max = x.max(dim=1, keepdim=True).values
        exp_x = torch.exp(self.r * (x - x_max))
        sum_exp = exp_x.mean(dim=1, keepdim=True)
        out = x_max + (1.0 / self.r) * torch.log(sum_exp)
        return out.squeeze(1)  # (B, C)


class SimPool2d(nn.Module):
    """SimPool: Simple Attention-Based Pooling for 2D (NCHW) inputs.

    From 'Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?'
    https://arxiv.org/abs/2309.06891

    Uses GAP as query initialization and applies cross-attention between the GAP query
    and spatial features to produce a weighted pooled representation.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 1,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            gamma: Optional[float] = None,
            norm_layer: Optional[Type[nn.Module]] = None,
            flatten: bool = False,
            device=None,
            dtype=None,
    ):
        """
        Args:
            dim: Input feature dimension (number of channels).
            num_heads: Number of attention heads.
            qkv_bias: If True, add bias to query and key projections.
            qk_norm: If True, apply normalization to queries and keys.
            gamma: If provided, apply power normalization to values with this exponent.
            norm_layer: Normalization layer for patches and optionally qk_norm.
            flatten: If True, flatten output to (B, C).
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gamma = gamma
        self.flatten = flatten
        self.fused_attn = use_fused_attn()

        norm_layer = norm_layer or nn.LayerNorm
        self.norm = norm_layer(dim, **dd)
        self.q = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        self.k = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim, **dd)
            self.k_norm = norm_layer(self.head_dim, **dd)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Reshape to (B, N, C) for attention
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # GAP as query initialization
        q = x.mean(dim=1, keepdim=True)  # (B, 1, C)

        # Normalize patches for keys and values
        x_norm = self.norm(x)

        # Project query and keys
        q = self.q(q).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = x_norm.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.gamma is not None:
            # Power normalization on values
            v_min = v.amin(dim=-2, keepdim=True)
            v_shifted = v - v_min + 1e-6
            if self.fused_attn:
                attn_out = F.scaled_dot_product_attention(q, k, v_shifted.pow(self.gamma))
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn_out = attn @ v_shifted.pow(self.gamma)
            out = attn_out.pow(1.0 / self.gamma)
        else:
            if self.fused_attn:
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                out = attn @ v

        # (B, num_heads, 1, head_dim) -> (B, C) or (B, 1, C)
        out = out.transpose(1, 2).reshape(B, 1, C)
        if self.flatten:
            out = out.squeeze(1)
        return out


class SimPool1d(nn.Module):
    """SimPool: Simple Attention-Based Pooling for sequence (NLC) inputs.

    From 'Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?'
    https://arxiv.org/abs/2309.06891

    Uses GAP as query initialization and applies cross-attention between the GAP query
    and sequence tokens to produce a weighted pooled representation.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 1,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            gamma: Optional[float] = None,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None,
    ):
        """
        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            qkv_bias: If True, add bias to query and key projections.
            qk_norm: If True, apply normalization to queries and keys.
            gamma: If provided, apply power normalization to values with this exponent.
            norm_layer: Normalization layer for tokens and optionally qk_norm.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gamma = gamma
        self.fused_attn = use_fused_attn()

        norm_layer = norm_layer or nn.LayerNorm
        self.norm = norm_layer(dim, **dd)
        self.q = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        self.k = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim, **dd)
            self.k_norm = norm_layer(self.head_dim, **dd)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # GAP as query initialization
        q = x.mean(dim=1, keepdim=True)  # (B, 1, C)

        # Normalize tokens for keys and values
        x_norm = self.norm(x)

        # Project query and keys
        q = self.q(q).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = x_norm.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.gamma is not None:
            # Power normalization on values
            v_min = v.amin(dim=-2, keepdim=True)
            v_shifted = v - v_min + 1e-6
            if self.fused_attn:
                attn_out = F.scaled_dot_product_attention(q, k, v_shifted.pow(self.gamma))
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn_out = attn @ v_shifted.pow(self.gamma)
            out = attn_out.pow(1.0 / self.gamma)
        else:
            if self.fused_attn:
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                out = attn @ v

        # (B, num_heads, 1, head_dim) -> (B, C)
        out = out.transpose(1, 2).reshape(B, C)
        return out
