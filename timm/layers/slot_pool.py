""" Slot Attention Pooling

Slot Attention mechanism adapted for use as a pooling layer.

Based on 'Object-Centric Learning with Slot Attention' by Locatello et al.
https://arxiv.org/abs/2006.15055

Original implementation reference:
https://github.com/lucidrains/slot-attention (MIT License)

Adapted for timm by Ross Wightman, original PR code by Bill Psomas
"""
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import use_fused_attn
from .mlp import Mlp


class SlotPool(nn.Module):
    """Slot Attention pooling module.

    Unlike standard attention pooling, Slot Attention uses iterative refinement
    with competition between slots. The softmax is applied over slots (not keys),
    causing slots to compete for explaining input locations.

    This creates a soft clustering effect where each slot specializes to different
    parts of the input, useful for object-centric representations.

    For standard pooling use cases, set num_slots=1 and iters=1 to get behavior
    closer to AttentionPoolLatent but with the slot attention update mechanism.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_slots: int = 1,
            iters: int = 3,
            hidden_dim: Optional[int] = None,
            mlp_ratio: float = 2.0,
            qkv_bias: bool = True,
            stochastic_init: bool = False,
            pool_type: str = 'max',
            eps: float = 1e-8,
            norm_layer: Optional[Type[nn.Module]] = None,
            act_layer: Optional[Type[nn.Module]] = nn.GELU,
            device=None,
            dtype=None,
    ):
        """
        Args:
            dim: Input feature dimension.
            num_slots: Number of slot vectors. For pooling, 1 is typical.
            iters: Number of iterative refinement steps.
            hidden_dim: Hidden dimension for slot MLP. Defaults to dim * mlp_ratio.
            mlp_ratio: Ratio for hidden dim if hidden_dim not specified.
            qkv_bias: If True, add bias to q, k, v projections.
            stochastic_init: If True, initialize slots with learned mu + learned sigma * noise.
                If False, slots are deterministically initialized from learned parameters.
            pool_type: How to aggregate multiple slots - 'max', 'avg', or 'first'.
            eps: Small constant for numerical stability in attention normalization.
            norm_layer: Normalization layer class.
            act_layer: Activation layer class for MLP.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.stochastic_init = stochastic_init
        self.pool_type = pool_type
        self.fused_attn = use_fused_attn()

        norm_layer = norm_layer or nn.LayerNorm

        # Slot initialization parameters
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, dim, **dd))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim, **dd))

        # Projections - separate q, k, v (no fused qkv since q comes from slots, kv from input)
        self.q = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        self.k = nn.Linear(dim, dim, bias=qkv_bias, **dd)
        self.v = nn.Linear(dim, dim, bias=qkv_bias, **dd)

        # GRU for slot updates
        self.gru = nn.GRUCell(dim, dim, **dd)

        # MLP after GRU update
        hidden_dim = hidden_dim or int(dim * mlp_ratio)
        self.norm_mlp = norm_layer(dim, **dd)
        self.mlp = Mlp(dim, hidden_dim, act_layer=act_layer, **dd)

        # Input normalization
        self.norm_input = norm_layer(dim, **dd)
        self.norm_slots = norm_layer(dim, **dd)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.slots_mu)
        if self.stochastic_init:
            nn.init.xavier_uniform_(self.slots_log_sigma)

    def _init_slots(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize slot vectors."""
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        if self.stochastic_init and self.training:
            sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            # Deterministic: just use mu repeated for each slot
            # Add small learned perturbation per slot to break symmetry
            slots = mu
        return slots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C) where N is sequence length.

        Returns:
            Pooled output of shape (B, C).
        """
        B, N, C = x.shape
        device, dtype = x.device, x.dtype

        # Initialize slots
        slots = self._init_slots(B, device, dtype)

        # Normalize input and compute k, v (constant across iterations)
        x = self.norm_input(x)
        k = self.k(x)  # (B, N, C)
        v = self.v(x)  # (B, N, C)

        # Iterative refinement
        for _ in range(self.iters):
            slots_prev = slots

            # Normalize slots and compute queries
            slots = self.norm_slots(slots)
            q = self.q(slots)  # (B, num_slots, C)

            # Compute attention: (B, num_slots, N)
            # Note: we do NOT use F.sdpa here because we need softmax over slots (dim=1),
            # not over keys (dim=-1) as standard attention does
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_slots, N)

            # Softmax over SLOTS (not keys) - this is the key difference from standard attention
            # Each input location decides which slot to route to
            attn = attn.softmax(dim=1)  # normalize over slots
            attn = attn + self.eps

            # Weighted mean over slots (normalize so weights sum to 1 per slot)
            attn = attn / attn.sum(dim=-1, keepdim=True)  # (B, num_slots, N)

            # Aggregate values into slots
            updates = attn @ v  # (B, num_slots, C)

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, C),
                slots_prev.reshape(B * self.num_slots, C),
            )
            slots = slots.reshape(B, self.num_slots, C)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Pool slots to single vector
        if self.pool_type == 'max':
            out = slots.max(dim=1).values
        elif self.pool_type == 'avg':
            out = slots.mean(dim=1)
        elif self.pool_type == 'first':
            out = slots[:, 0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        return out


class SlotPool2d(nn.Module):
    """Slot Attention pooling for 2D (NCHW) inputs.

    Convenience wrapper that handles NCHW -> NLC conversion.
    """
    def __init__(
            self,
            dim: int,
            num_slots: int = 1,
            iters: int = 3,
            hidden_dim: Optional[int] = None,
            mlp_ratio: float = 2.0,
            qkv_bias: bool = True,
            stochastic_init: bool = False,
            pool_type: str = 'max',
            eps: float = 1e-8,
            norm_layer: Optional[Type[nn.Module]] = None,
            act_layer: Optional[Type[nn.Module]] = nn.GELU,
            flatten: bool = True,
            device=None,
            dtype=None,
    ):
        """
        Args:
            dim: Input feature dimension (channels).
            num_slots: Number of slot vectors.
            iters: Number of iterative refinement steps.
            hidden_dim: Hidden dimension for slot MLP.
            mlp_ratio: Ratio for hidden dim if hidden_dim not specified.
            qkv_bias: If True, add bias to q, k, v projections.
            stochastic_init: If True, use stochastic slot initialization during training.
            pool_type: How to aggregate multiple slots - 'max', 'avg', or 'first'.
            eps: Small constant for numerical stability.
            norm_layer: Normalization layer class.
            act_layer: Activation layer class for MLP.
            flatten: If True, output shape is (B, C). If False, (B, 1, C).
        """
        super().__init__()
        self.flatten = flatten
        self.pool = SlotPool(
            dim=dim,
            num_slots=num_slots,
            iters=iters,
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            stochastic_init=stochastic_init,
            pool_type=pool_type,
            eps=eps,
            norm_layer=norm_layer,
            act_layer=act_layer,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Pooled output of shape (B, C) if flatten=True, else (B, 1, C).
        """
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        out = self.pool(x)  # (B, C)
        if not self.flatten:
            out = out.unsqueeze(1)
        return out
