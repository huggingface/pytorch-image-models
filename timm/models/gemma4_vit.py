"""Gemma4 Vision Transformer

Vision encoder from Google's Gemma 4 multimodal model.
Custom ViT with 2D RoPE, Gated MLP, QKV normalization, and 4-norm sandwich blocks.

Paper: https://ai.google.dev/gemma/docs/core/model_card_4
Reference impl: https://github.com/huggingface/transformers (Gemma4VisionModel)

Copyright 2025 Yonghye Kwon
"""

import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, trunc_normal_tf_, use_fused_attn

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint, named_apply
from ._registry import generate_default_cfgs, register_model

__all__ = ['Gemma4Vit']


class Gemma4ClippableLinear(nn.Module):
    """Linear layer with optional input/output clamping.

    Used in Gemma4 E4B variant where clamp values are finite and affect output.
    When use_clipped=False, behaves as a standard nn.Linear (no buffers registered).
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_clipped: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.use_clipped = use_clipped
        self.linear = nn.Linear(in_features, out_features, bias=False, **dd)
        if use_clipped:
            self.register_buffer('input_min', torch.empty((), **dd))
            self.register_buffer('input_max', torch.empty((), **dd))
            self.register_buffer('output_min', torch.empty((), **dd))
            self.register_buffer('output_max', torch.empty((), **dd))

            self.reset_parameters()

    def reset_parameters(self) -> None:
        # ``nn.Linear`` handles its own weight init; only clamp buffers need resetting.
        # Default to no-op clamp (±inf); pretrained checkpoints overwrite these.
        if self.use_clipped:
            self.input_min.fill_(-float('inf'))
            self.input_max.fill_(float('inf'))
            self.output_min.fill_(-float('inf'))
            self.output_max.fill_(float('inf'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipped:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipped:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


class Gemma4RMSNorm(nn.RMSNorm):
    """RMSNorm with optional learnable scale.

    Thin wrapper over ``nn.RMSNorm`` that renames ``elementwise_affine`` to
    ``with_scale`` (HF Gemma4 omits the V-norm gain so we need a way to
    request the no-scale variant).
    """

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            with_scale: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=with_scale, device=device, dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Apply rotary position embedding to input tensor."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    ndim: int = 2,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Apply multidimensional RoPE to input tensor.

    Splits input along head_dim into ndim parts, applies RoPE to each,
    then concatenates back.
    """
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    split_sizes = [num_rotated_channels_per_dim] * ndim

    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)

    y_parts = [
        apply_rotary_pos_emb(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim=unsqueeze_dim) for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


class Gemma4RotaryEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for Gemma4 vision encoder.

    Computes RoPE independently for each spatial dimension (x, y),
    using theta=100.0 and the partitioned head_dim.
    """

    def __init__(
            self,
            head_dim: int,
            rope_theta: float = 100.0,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        # Each spatial dimension uses head_dim//2 channels, so head_dim//4 frequencies per dimension.
        # ``inv_freq`` is always kept in float32 regardless of module dtype.
        num_freqs = (head_dim // 2) // 2
        self.register_buffer(
            'inv_freq',
            torch.empty(num_freqs, device=device, dtype=torch.float),
            persistent=False,
        )

        self._init_buffers()

    def _init_buffers(self) -> None:
        """Compute and fill non-persistent buffer values."""
        spatial_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.float, device=self.inv_freq.device) / spatial_dim
            )
        )
        self.inv_freq.copy_(inv_freq)

    def reset_parameters(self) -> None:
        self._init_buffers()

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Hidden states tensor, used only for dtype/device.
            position_ids: (B, N, 2) tensor of (x, y) patch coordinates.

        Returns:
            cos, sin: (B, N, head_dim) tensors for RoPE application.
        """
        # Use a ``with`` block rather than a ``@torch.no_grad()`` decorator so that
        # ``torch.jit.script`` sees the original method body instead of the
        # ``contextlib`` wrapper (which otherwise fails with ``undefined value torch``).
        with torch.no_grad():
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

            all_cos: List[torch.Tensor] = []
            all_sin: List[torch.Tensor] = []
            for i in range(2):  # x and y dimensions
                dim_pos = position_ids[:, :, i]  # (B, N)
                dim_pos_expanded = dim_pos[:, None, :].float()  # (B, 1, N)

                freqs = (inv_freq_expanded.float() @ dim_pos_expanded.float()).transpose(1, 2)  # (B, N, spatial_dim//2)
                emb = torch.cat((freqs, freqs), dim=-1)  # (B, N, spatial_dim)
                all_cos.append(emb.cos())
                all_sin.append(emb.sin())

            cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)  # (B, N, head_dim)
            sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


class Gemma4PatchEmbed(nn.Module):
    """Linear patch embedding with learned 2D position embedding table.

    Unlike standard ViT PatchEmbed (conv2d), Gemma4 uses:
    - Linear projection on flattened patches
    - Separate 2D position embedding table with one-hot lookup
    """

    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            position_embedding_size: int = 10240,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.position_embedding_size = position_embedding_size

        self.input_proj = nn.Linear(in_chans * patch_size**2, embed_dim, bias=False, **dd)
        self.position_embedding_table = nn.Parameter(
            torch.empty(2, position_embedding_size, embed_dim, **dd)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_tf_(self.position_embedding_table, std=.02)

    def _extract_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract and flatten patches from (B, C, H, W) image tensor."""
        B, C, H, W = pixel_values.shape
        pH = H // self.patch_size
        pW = W // self.patch_size
        x = pixel_values.reshape(B, C, pH, self.patch_size, pW, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, pH, pW, C, P, P)
        x = x.reshape(B, pH * pW, C * self.patch_size * self.patch_size)
        return x

    def _position_embeddings(
        self,
        position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute position embeddings from 2D position table using one-hot matmul.

        ``position_ids`` follows the Gemma4-internal ``(x, y)`` convention.
        """
        clamped_positions = position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        # (B, 2, N, pos_size) @ (2, pos_size, embed_dim) -> (B, 2, N, embed_dim)
        position_embeddings = one_hot @ self.position_embedding_table
        # Sum across x and y dimensions
        position_embeddings = position_embeddings.sum(dim=1)  # (B, N, embed_dim)
        # Zero out padding positions
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
        return position_embeddings

    def _project_patches(
        self,
        patches: torch.Tensor,
        position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Project pre-extracted patches and add 2D position embeddings.

        Args:
            patches: (B, N, C*P*P) patches in Gemma4-internal ``C-P-P`` flatten order.
            position_ids: (B, N, 2) patch coordinates in Gemma4-internal ``(x, y)`` convention.
            padding_positions: (B, N) boolean mask, True for padding tokens.
        """
        # Scale to [-1, 1] (matching HF's `2 * (pixel_values - 0.5)`)
        patches = 2 * (patches - 0.5)
        hidden_states = self.input_proj(patches.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(position_ids, padding_positions)
        return hidden_states + position_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Raw-image patch-embedding entry point (kept for the ``(B, C, H, W)`` path).

        Args:
            pixel_values: (B, C, H, W) image tensor, normalized to [0, 1].
            position_ids: (B, N, 2) patch position coordinates (Gemma4-internal ``(x, y)``).
            padding_positions: (B, N) boolean mask, True for padding.

        Returns:
            (B, N, embed_dim) patch embeddings with position information.
        """
        patches = self._extract_patches(pixel_values)
        return self._project_patches(patches, position_ids, padding_positions)


class Gemma4Attention(nn.Module):
    """Gemma4 Vision Attention with QKV normalization and 2D RoPE.

    Key features:
    - Separate Q, K, V projections (not fused)
    - RMSNorm on Q, K (with scale) and V (without scale)
    - 2D RoPE applied after normalization
    - Attention scale = 1.0 (since QK are normalized)
    """

    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 12,
            head_dim: int = 64,
            num_kv_heads: Optional[int] = None,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_eps: float = 1e-6,
            use_clipped_linears: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.fused_attn = use_fused_attn()

        self.q_proj = Gemma4ClippableLinear(dim, num_heads * head_dim, use_clipped=use_clipped_linears, **dd)
        self.k_proj = Gemma4ClippableLinear(dim, self.num_kv_heads * head_dim, use_clipped=use_clipped_linears, **dd)
        self.v_proj = Gemma4ClippableLinear(dim, self.num_kv_heads * head_dim, use_clipped=use_clipped_linears, **dd)
        self.o_proj = Gemma4ClippableLinear(num_heads * head_dim, dim, use_clipped=use_clipped_linears, **dd)

        self.q_norm = Gemma4RMSNorm(head_dim, eps=norm_eps, with_scale=True, **dd)
        self.k_norm = Gemma4RMSNorm(head_dim, eps=norm_eps, with_scale=True, **dd)
        self.v_norm = Gemma4RMSNorm(head_dim, eps=norm_eps, with_scale=False, **dd)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim)

        # Normalize Q, K, V
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Apply 2D RoPE to Q and K
        q = apply_multidimensional_rope(q, rope_cos, rope_sin, ndim=2, unsqueeze_dim=2)
        k = apply_multidimensional_rope(k, rope_cos, rope_sin, ndim=2, unsqueeze_dim=2)

        # Transpose to (B, heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=1.0,
            )
        else:
            # Manual attention with scale=1.0
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.o_proj(x)
        x = self.proj_drop(x)
        return x


class Gemma4GatedMlp(nn.Module):
    """Gated MLP for Gemma4 Vision Encoder.

    Uses GELUTanh activation: output = down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))
    """

    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            act_layer: Optional[Callable] = None,
            drop: float = 0.0,
            use_clipped_linears: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.gate_proj = Gemma4ClippableLinear(in_features, hidden_features, use_clipped=use_clipped_linears, **dd)
        self.up_proj = Gemma4ClippableLinear(in_features, hidden_features, use_clipped=use_clipped_linears, **dd)
        self.down_proj = Gemma4ClippableLinear(hidden_features, in_features, use_clipped=use_clipped_linears, **dd)
        self.act = act_layer() if act_layer is not None else nn.GELU(approximate='tanh')
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x)))


class Gemma4Block(nn.Module):
    """Gemma4 Vision Encoder Block with 4-norm sandwich pattern.

    Unlike standard ViT (pre-norm with 2 norms), Gemma4 uses:
    - input_layernorm (norm1) + post_attention_layernorm (norm2)
    - pre_feedforward_layernorm (norm3) + post_feedforward_layernorm (norm4)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: int,
            intermediate_size: int,
            num_kv_heads: Optional[int] = None,
            norm_eps: float = 1e-6,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: Optional[Callable] = None,
            use_clipped_linears: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.norm1 = Gemma4RMSNorm(dim, eps=norm_eps, **dd)
        self.attn = Gemma4Attention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_eps=norm_eps,
            use_clipped_linears=use_clipped_linears,
            **dd,
        )
        self.norm2 = Gemma4RMSNorm(dim, eps=norm_eps, **dd)
        self.norm3 = Gemma4RMSNorm(dim, eps=norm_eps, **dd)
        self.mlp = Gemma4GatedMlp(
            in_features=dim,
            hidden_features=intermediate_size,
            act_layer=act_layer,
            use_clipped_linears=use_clipped_linears,
            **dd,
        )
        self.norm4 = Gemma4RMSNorm(dim, eps=norm_eps, **dd)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with sandwich norm
        residual = x
        x = self.norm1(x)
        x = self.attn(x, rope_cos, rope_sin, position_ids, attn_mask=attn_mask)
        x = self.norm2(x)
        x = residual + self.drop_path(x)

        # MLP with sandwich norm
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = self.norm4(x)
        x = residual + self.drop_path(x)
        return x


class Gemma4VisionPooler(nn.Module):
    """Spatial pooling for Gemma4 vision encoder output.

    Pools patches by averaging within k×k grid cells based on position coordinates.
    Output is scaled by sqrt(hidden_size).
    """

    def __init__(self, hidden_size: int, pooling_kernel_size: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.root_hidden_size = hidden_size**0.5
        self.pooling_kernel_size = pooling_kernel_size

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """2D spatial pooling on a ``k × k`` grid using ``self.pooling_kernel_size``.

        ``position_ids`` follows the Gemma4-internal ``(x, y)`` convention. Caller
        guarantees both grid dimensions divide by ``k``; we assert ``k^2 * output_length == N``
        so a mis-sized grid fails loudly instead of silently aliasing.
        """
        N = hidden_states.shape[1]
        k = self.pooling_kernel_size
        k_squared = k * k
        if k_squared * output_length != N:
            raise ValueError(
                f"Cannot pool {N} tokens into {output_length} with k={k}: "
                f"k^2 * output_length ({k_squared * output_length}) must equal N ({N}). "
                f"Both grid dimensions must be divisible by k."
            )

        # Per-sample grid width (max x-coord + 1); used as the y-stride when flattening (x, y) -> index.
        clamped_positions = position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode='floor')
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]

        weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, N, D) encoder output.
            position_ids: (B, N, 2) patch coordinates in Gemma4-internal ``(x, y)`` convention.
            padding_positions: (B, N) boolean mask, True for padding.
            output_length: Number of output tokens after pooling.

        Returns:
            Tuple of:
            - pooled hidden states (B, output_length, D)
            - pooler mask (B, output_length) True for valid tokens
        """
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        if hidden_states.shape[1] != output_length:
            hidden_states, pooler_mask = self._avg_pool_by_positions(
                hidden_states,
                position_ids,
                output_length,
            )
        else:
            pooler_mask = ~padding_positions

        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, pooler_mask


class Gemma4Vit(nn.Module):
    """Gemma4 Vision Transformer.

    A custom ViT used as the image encoder in Google's Gemma 4 multimodal model.
    Features 2D RoPE, gated MLP, QKV normalization, and 4-norm sandwich blocks.
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 768,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 0,
            global_pool: str = 'gemma4',
            embed_dim: int = 768,
            depth: int = 16,
            num_heads: int = 12,
            head_dim: int = 64,
            num_kv_heads: Optional[int] = None,
            intermediate_size: int = 3072,
            norm_eps: float = 1e-6,
            rope_theta: float = 100.0,
            position_embedding_size: int = 10240,
            pooling_kernel_size: int = 3,
            standardize: bool = False,
            use_clipped_linears: bool = False,
            drop_rate: float = 0.0,
            proj_drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            act_layer: Optional[Callable] = None,
            weight_init: str = '',
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = 0
        self.grad_checkpointing = False
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size
        self.use_clipped_linears = use_clipped_linears

        act_layer = act_layer or partial(nn.GELU, approximate='tanh')

        # Patch embedding
        self.patch_embed = Gemma4PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            position_embedding_size=position_embedding_size,
            **dd,
        )

        # 2D RoPE
        self.rotary_emb = Gemma4RotaryEmbedding2D(
            head_dim=head_dim,
            rope_theta=rope_theta,
            **dd,
        )

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Gemma4Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_kv_heads=num_kv_heads,
                    intermediate_size=intermediate_size,
                    norm_eps=norm_eps,
                    attn_drop=attn_drop_rate,
                    proj_drop=proj_drop_rate,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    use_clipped_linears=use_clipped_linears,
                    **dd,
                )
                for i in range(depth)
            ]
        )

        # Pooler
        self.pooler = Gemma4VisionPooler(
            hidden_size=embed_dim,
            pooling_kernel_size=pooling_kernel_size,
        )

        # Optional standardization buffers (used in 31B variant).
        # Values are set in ``init_weights`` (no-op transform by default;
        # pretrained checkpoints overwrite these).
        if standardize:
            self.register_buffer('std_bias', torch.empty(embed_dim, **dd))
            self.register_buffer('std_scale', torch.empty(embed_dim, **dd))
        else:
            self.std_bias = None
            self.std_scale = None

        # Classification head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

        # Feature info for intermediate extraction
        self.feature_info = [dict(num_chs=embed_dim, reduction=patch_size, module=f'blocks.{i}') for i in range(depth)]

        self.weight_init_mode = 'reset' if weight_init == 'skip' else weight_init
        # TODO: skip init when on meta device when safe to do so
        if weight_init != 'skip':
            self.init_weights(needs_reset=False)

    @torch.jit.ignore
    def init_weights(self, mode: str = '', needs_reset: bool = True) -> None:
        """Initialize model weights.

        Args:
            mode: Init mode. '' applies trunc-normal-TF Linear init; 'reset'
                only calls ``reset_parameters`` on each sub-module.
            needs_reset: If True, call ``reset_parameters`` on modules that
                have one (for post-``to_empty()`` reinit). Set to False during
                ``__init__`` since modules already self-initialize there.
        """
        mode = mode or self.weight_init_mode
        assert mode in ('', 'reset')

        # Top-level standardization buffers default to a no-op transform;
        # pretrained checkpoints overwrite these.
        if self.std_bias is not None:
            nn.init.zeros_(self.std_bias)
        if self.std_scale is not None:
            nn.init.ones_(self.std_scale)

        named_apply(get_init_weights_gemma4_vit(mode, needs_reset=needs_reset), self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'patch_embed.position_embedding_table'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^patch_embed|^rotary_emb',
            blocks=[(r'^blocks\.(\d+)', None), (r'^pooler|^std_', (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _default_patch_coord(
        self,
        batch_size: int,
        pH: int,
        pW: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create default grid patch coords + valid mask in NaFlex external (y, x) convention."""
        ys = torch.arange(pH, device=device)
        xs = torch.arange(pW, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        patch_coord = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)  # (N, 2) (y, x)
        patch_coord = patch_coord.unsqueeze(0).expand(batch_size, -1, -1)
        patch_valid = torch.ones(batch_size, pH * pW, dtype=torch.bool, device=device)
        return patch_coord, patch_valid

    def _naflex_to_internal_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Convert NaFlex pre-patched layout to Gemma4 internal C-P-P flat order.

        Accepts ``(B, N, P*P*C)`` (NaFlex P-P-C flat) or ``(B, N, Ph, Pw, C)`` and
        returns ``(B, N, C*P*P)`` in Gemma4's C-P-P flat order, ready for ``input_proj``.
        """
        P = self.patch_size
        if x.ndim == 5:
            B, N, Ph, Pw, C = x.shape
            return x.permute(0, 1, 4, 2, 3).reshape(B, N, C * Ph * Pw)
        if x.ndim == 3:
            B, N, PPC = x.shape
            C = PPC // (P * P)
            return x.reshape(B, N, P, P, C).permute(0, 1, 4, 2, 3).reshape(B, N, C * P * P)
        raise ValueError(f"Pre-patchified input must have ndim in (3, 5); got {x.ndim}")

    def _resolve_inputs(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        patch_coord: Optional[torch.Tensor],
        patch_valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize inputs from NaFlex (dict or tensor + coords) to internal form.

        Returns:
            x: tensor (raw image or pre-patched), with ``dict`` unwrapped.
            position_ids: (B, N, 2) Gemma4-internal (x, y) coords.
            padding_positions: (B, N) True for padding tokens.
            patch_coord: (B, N, 2) external NaFlex (y, x) coords (preserved for downstream).
            patch_valid: (B, N) True for valid tokens.
        """
        if isinstance(x, dict):
            patch_coord = x.get('patch_coord', patch_coord)
            patch_valid = x.get('patch_valid', patch_valid)
            x = x['patches']

        if x.ndim == 4 and patch_coord is None:
            B, _, H, W = x.shape
            P = self.patch_size
            # When using the built-in gemma4 spatial pooler, both image dims must
            # divide cleanly into (patch_size * pooling_kernel_size)-sized cells.
            # Pre-patchified / NaFlex inputs are assumed to already be conformant.
            if self.global_pool == 'gemma4':
                cell = P * self.pooling_kernel_size
                if H % cell != 0 or W % cell != 0:
                    raise ValueError(
                        f"Image size ({H}, {W}) must be divisible by "
                        f"patch_size * pooling_kernel_size ({cell}) when global_pool='gemma4'. "
                        f"Resize to multiples of {cell}, or use global_pool='avg'."
                    )
            pH = H // P
            pW = W // P
            patch_coord, patch_valid = self._default_patch_coord(B, pH, pW, x.device)

        if patch_coord is None:
            raise ValueError("patch_coord is required for pre-patchified input.")

        if patch_valid is None:
            sentinel = (patch_coord == -1).all(dim=-1)
            if sentinel.any():
                patch_valid = ~sentinel
            else:
                patch_valid = torch.ones(patch_coord.shape[:2], dtype=torch.bool, device=patch_coord.device)

        # External NaFlex (y, x) → Gemma4 internal (x, y)
        position_ids = patch_coord.flip(dims=(-1,))
        padding_positions = ~patch_valid
        return x, position_ids, padding_positions, patch_coord, patch_valid

    def _embed_and_encode(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        block_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        max_block_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Shared patch-embed + RoPE + encoder-block pipeline.

        ``x`` may be either a raw image ``(B, C, H, W)`` or a pre-patched tensor
        (``(B, N, P*P*C)`` / ``(B, N, Ph, Pw, C)``).
        """
        if x.ndim == 4:
            x = self.patch_embed(x, position_ids, padding_positions)
        else:
            patches = self._naflex_to_internal_patches(x)
            x = self.patch_embed._project_patches(patches, position_ids, padding_positions)

        B, N = x.shape[:2]
        rope_cos, rope_sin = self.rotary_emb(x, position_ids)

        attn_mask: Optional[torch.Tensor] = None
        if padding_positions.any():
            # Column-only additive mask broadcast over heads (B, 1, 1, N) -> (B, heads, q, N).
            # Matches HF Gemma4 implementation and preserves bit-perfect equivalence.
            attn_mask = torch.zeros(B, 1, 1, N, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(padding_positions[:, None, None, :], float('-inf'))

        if max_block_index is None:
            blocks = self.blocks
        else:
            blocks = self.blocks[: max_block_index + 1]

        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope_cos, rope_sin, position_ids, attn_mask)
            else:
                x = blk(x, rope_cos, rope_sin, position_ids, attn_mask=attn_mask)
            if block_callback is not None:
                block_callback(i, x)

        return x

    def forward_features(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, position_ids, padding_positions, _, _ = self._resolve_inputs(x, patch_coord, patch_valid)
        return self._embed_and_encode(x, position_ids, padding_positions)

    def forward_head(
        self,
        x: torch.Tensor,
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        pre_logits: bool = False,
    ) -> torch.Tensor:
        position_ids = patch_coord.flip(dims=(-1,)) if patch_coord is not None else None
        padding_positions = ~patch_valid if patch_valid is not None else None

        if self.global_pool == 'gemma4' and position_ids is not None:
            if padding_positions is None:
                padding_positions = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
            output_length = x.shape[1] // (self.pooling_kernel_size ** 2)
            x, _ = self.pooler(x, position_ids, padding_positions, output_length)
            if self.std_bias is not None:
                x = (x - self.std_bias) * self.std_scale
        elif self.global_pool == 'avg':
            if padding_positions is not None:
                x = x.masked_fill(padding_positions.unsqueeze(-1), 0.0)
                x = x.sum(dim=1) / (~padding_positions).sum(dim=1, keepdim=True).clamp(min=1)
            else:
                x = x.mean(dim=1)

        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, position_ids, padding_positions, patch_coord, patch_valid = self._resolve_inputs(
            x, patch_coord, patch_valid,
        )
        features = self._embed_and_encode(x, position_ids, padding_positions)
        return self.forward_head(features, patch_coord=patch_coord, patch_valid=patch_valid)

    def forward_intermediates(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        indices: Optional[Union[int, List[int]]] = None,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHW',
        intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features returning intermediates.

        Args:
            x: Input tensor ``(B, C, H, W)`` or NaFlex pre-patchified tensor/dict.
            patch_coord: ``(B, N, 2)`` patch coords in NaFlex external (y, x) convention.
            patch_valid: ``(B, N)`` boolean mask, True for valid tokens.
            indices: Block indices to return intermediates for.
            norm: Not used (no final norm in Gemma4 encoder).
            stop_early: Stop iterating after last needed intermediate.
            output_fmt: Output format ('NCHW' or 'NLC'). NCHW requires a fixed full grid.
            intermediates_only: Only return intermediate features.
        """
        assert output_fmt in ('NCHW', 'NLC')
        reshape = output_fmt == 'NCHW'
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        x, position_ids, padding_positions, _, _ = self._resolve_inputs(x, patch_coord, patch_valid)

        intermediates: List[torch.Tensor] = []
        raw_input_ndim = x.ndim

        def _cb(i: int, y: torch.Tensor) -> None:
            if i in take_indices:
                intermediates.append(y)

        max_block_index = None
        if stop_early and not torch.jit.is_scripting():
            max_block_index = max_index

        x = self._embed_and_encode(
            x,
            position_ids,
            padding_positions,
            block_callback=_cb,
            max_block_index=max_block_index,
        )

        if reshape:
            if raw_input_ndim != 4:
                raise ValueError("output_fmt='NCHW' requires a raw image (B, C, H, W) input.")
            # Recover grid from internal position_ids (x, y) max.
            B = position_ids.shape[0]
            pW = int(position_ids[..., 0].max().item()) + 1
            pH = int(position_ids[..., 1].max().item()) + 1
            intermediates = [y.reshape(B, pH, pW, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
        self,
        indices: Union[int, List[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ) -> List[int]:
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[: max_index + 1]
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices


def init_weights_gemma4_vit(module: nn.Module, name: str = '', needs_reset: bool = True) -> None:
    """Per-module init for Gemma4Vit (trunc-normal-TF for Linear weights).

    Args:
        module: Module to initialize.
        name: Dotted module name (from ``named_apply``).
        needs_reset: If True, call ``reset_parameters`` on modules that define one.
    """
    if isinstance(module, nn.Linear):
        trunc_normal_tf_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
    elif needs_reset and hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def get_init_weights_gemma4_vit(mode: str = '', needs_reset: bool = True) -> Callable:
    # Only the default trunc-normal scheme; 'reset' is handled inside the fn.
    return partial(init_weights_gemma4_vit, needs_reset=needs_reset)


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: Gemma4Vit) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace Transformers Gemma4 vision encoder weights to timm format.

    Handles key remapping from HF naming conventions to timm naming conventions.
    Supports both `model.vision_tower.` (actual) and `model.vision_model.` (docs) prefixes.
    Preserves ClippableLinear structure (.linear.weight and clamp buffers) for E4B variant.
    """
    out_dict = {}

    # Detect prefix from keys
    hf_prefixes = ('model.vision_tower.', 'model.vision_model.', 'vision_model.', 'vision_tower.')

    for k, v in state_dict.items():
        # Check if this is a vision key
        matched_prefix = None
        for prefix in hf_prefixes:
            if k.startswith(prefix):
                matched_prefix = prefix
                break

        if matched_prefix is None:
            # Already in timm format — pass through
            if k.startswith(('patch_embed.', 'blocks.', 'std_', 'head.', 'pooler.', 'rotary_emb.')):
                out_dict[k] = v
            continue

        # Strip HF prefix
        new_k = k[len(matched_prefix) :]

        # Skip rotary embedding buffers (recomputed in model)
        if 'rotary_emb' in new_k:
            continue

        # Remap patch embedder
        new_k = new_k.replace('patch_embedder.', 'patch_embed.')

        # Remap encoder layers
        new_k = new_k.replace('encoder.layers.', 'blocks.')

        # Remap norm layers
        new_k = new_k.replace('.input_layernorm.', '.norm1.')
        new_k = new_k.replace('.post_attention_layernorm.', '.norm2.')
        new_k = new_k.replace('.pre_feedforward_layernorm.', '.norm3.')
        new_k = new_k.replace('.post_feedforward_layernorm.', '.norm4.')

        # Remap attention
        new_k = new_k.replace('.self_attn.', '.attn.')

        # Note: .linear.weight and clamp buffers (input_min etc.) are preserved as-is
        # since our model uses Gemma4ClippableLinear which has the same structure.

        out_dict[new_k] = v

    return out_dict


def _create_gemma4_vit(variant: str, pretrained: bool = False, **kwargs) -> Gemma4Vit:
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        Gemma4Vit,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 0,
        'input_size': (3, 768, 768),
        'pool_size': None,
        'crop_pct': 1.0,
        'interpolation': 'bicubic',
        'fixed_input_size': False,
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.input_proj',
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        'gemma4_vit_e4b.gemma4_e4b': _cfg(
            hf_hub_id='developer0hye/gemma4_vit_e4b',
            license='apache-2.0',
        ),
        'gemma4_vit_31b.gemma4_31b': _cfg(
            hf_hub_id='developer0hye/gemma4_vit_31b',
            license='apache-2.0',
        ),
    }
)


@register_model
def gemma4_vit_e4b(pretrained: bool = False, **kwargs) -> Gemma4Vit:
    """Gemma4 Vision Encoder from the E4B (4B) model variant (~150M params)."""
    model_args = dict(
        embed_dim=768,
        depth=16,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        standardize=False,
        use_clipped_linears=True,
    )
    return _create_gemma4_vit('gemma4_vit_e4b', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def gemma4_vit_31b(pretrained: bool = False, **kwargs) -> Gemma4Vit:
    """Gemma4 Vision Encoder from the 31B model variant (~550M params)."""
    model_args = dict(
        embed_dim=1152,
        depth=27,
        num_heads=16,
        head_dim=72,
        intermediate_size=4304,
        standardize=True,
    )
    return _create_gemma4_vit('gemma4_vit_31b', pretrained=pretrained, **dict(model_args, **kwargs))
