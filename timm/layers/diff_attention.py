"""Differential Attention

Paper: 'Differential Transformer' - https://arxiv.org/abs/2410.05258

Reference impl: https://github.com/microsoft/unilm/tree/master/Diff-Transformer

Hacked together by / Copyright 2024, Ross Wightman
"""
import math
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import maybe_add_mask, resolve_self_attn_mask
from .config import use_fused_attn
from .norm import RmsNorm


class DiffAttention(nn.Module):
    """Differential Attention module.

    Computes attention as the difference between two softmax attention maps, which helps
    cancel out noise and promotes sparse attention patterns. The module splits Q and K
    into two groups, computes separate attention maps, and subtracts one from the other
    scaled by a learnable lambda parameter.

    The attention output is computed as:
        Attn = softmax(Q1 @ K1^T) - lambda * softmax(Q2 @ K2^T)
        Output = Attn @ V

    Supports both fused (scaled_dot_product_attention) and manual implementations.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
            depth: int = 0,
            dual_lambda: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize the DiffAttention module.

        Args:
            dim: Input dimension of the token embeddings.
            num_heads: Number of attention heads.
            qkv_bias: Whether to use bias in the query, key, value projections.
            qk_norm: Whether to apply normalization to query and key vectors.
            scale_norm: Whether to apply normalization before the output projection.
            proj_bias: Whether to use bias in the output projection.
            attn_drop: Dropout rate applied to the attention weights.
            proj_drop: Dropout rate applied after the output projection.
            norm_layer: Normalization layer constructor (defaults to RmsNorm).
            depth: Block depth index, used to compute depth-dependent lambda_init.
            dual_lambda: If True, use simplified dual scalar lambda parameterization
                (2 params). If False, use the paper's original formulation with
                lambda_q/k vectors (4 * head_dim params).
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if norm_layer is None:
            norm_layer = RmsNorm
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop
        self.norm = norm_layer(dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dual_lambda = dual_lambda
        if dual_lambda:
            self.lambda_a = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
            self.lambda_b = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
            self.lambda_q1 = self.lambda_k1 = self.lambda_q2 = self.lambda_k2 = None
        else:
            self.lambda_a = self.lambda_b = None
            self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))

        self.sub_norm = RmsNorm(2 * self.head_dim, eps=1e-5, **dd)

        self.lambda_init = 0.8
        self.set_lambda_init(depth)
        self.reset_parameters()

    def set_lambda_init(self, depth: int):
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

    def reset_parameters(self):
        if self.dual_lambda:
            nn.init.zeros_(self.lambda_a)
            nn.init.zeros_(self.lambda_b)
        else:
            nn.init.normal_(self.lambda_q1, mean=0, std=0.1)
            nn.init.normal_(self.lambda_k1, mean=0, std=0.1)
            nn.init.normal_(self.lambda_q2, mean=0, std=0.1)
            nn.init.normal_(self.lambda_k2, mean=0, std=0.1)

    def _compute_lambda(self) -> torch.Tensor:
        if self.lambda_a is not None:
            lambda_1 = torch.exp(self.lambda_a)
            lambda_2 = torch.exp(self.lambda_b)
        else:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        return lambda_1 - lambda_2 + self.lambda_init

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=2)
        q = q.reshape(B, N, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, 2 * self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        lambda_full = self._compute_lambda().type_as(q)

        if self.fused_attn:
            q = q.reshape(B, self.num_heads, 2, N, self.head_dim)
            k = k.reshape(B, self.num_heads, 2, N, self.head_dim)
            q1, q2 = q.unbind(2)
            k1, k2 = k.unbind(2)

            dropout_p = self.attn_drop_p if self.training else 0.0
            attn1 = F.scaled_dot_product_attention(
                q1, k1, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            attn2 = F.scaled_dot_product_attention(
                q2, k2, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

            x = attn1 - lambda_full * attn2
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal=is_causal)
            attn = maybe_add_mask(attn, attn_bias)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            attn = attn.view(B, self.num_heads, 2, N, N)
            attn = attn[:, :, 0] - lambda_full * attn[:, :, 1]
            x = attn @ v

        x = self.sub_norm(x)
        x = x * (1 - self.lambda_init)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
