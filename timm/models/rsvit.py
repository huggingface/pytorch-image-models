""" Recursive Supervision Vision Transformers

A WIP recursive transformers that use shared blocks with recursive supervision
during training and adaptive halting during inference.

Key features:
- Recursive computation with shared blocks (z_block for latents / patches, y_block for class / output token)
- Recursive supervision at multiple steps during training
- Adaptive halting based on confidence during inference
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    PatchEmbed,
    RmsNorm,
    SwiGLU,
    AttentionRope,
    create_rope_embed,
    DropPath,
    LayerScale,
    trunc_normal_,
    use_fused_attn,
    apply_rot_embed_cat,
)
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model

__all__ = ['RSViT', 'RSPViT', 'RSTViT', 'ResidualMLP']

_logger = logging.getLogger(__name__)


class CrossAttentionRope(nn.Module):
    """Cross-attention module with RoPE support.

    Q comes from one source (e.g., latents/cls), KV from another (e.g., patches).
    RoPE is applied to K (context has spatial positions), Q optionally.

    Args:
        dim: Input dimension for queries.
        dim_context: Input dimension for context (keys/values). If None, same as dim.
        num_heads: Number of attention heads.
        dim_out: Output dimension. If None, same as dim.
        qkv_bias: Whether to use bias in Q, K, V projections.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
        attn_head_dim: Dimension per head. If None, computed as dim // num_heads.
        proj_bias: Whether to use bias in output projection.
        rotate_half: Use 'half' RoPE layout instead of 'interleaved'.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            dim_context: Optional[int] = None,
            num_heads: int = 8,
            dim_out: Optional[int] = None,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            proj_bias: bool = False,
            rotate_half: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        dim_context = dim_context or dim
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            head_dim = dim // num_heads

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.rotate_half = rotate_half

        self.q_proj = nn.Linear(dim, self.attn_dim, bias=qkv_bias, **dd)
        self.k_proj = nn.Linear(dim_context, self.attn_dim, bias=qkv_bias, **dd)
        self.v_proj = nn.Linear(dim_context, self.attn_dim, bias=qkv_bias, **dd)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attn_dim, dim_out, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            rope_q: Optional[torch.Tensor] = None,
            rope_k: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query input [B, N_q, D].
            context: Key/value input [B, N_kv, D_context].
            rope_q: RoPE embeddings for query positions.
            rope_k: RoPE embeddings for key/context positions.
            attn_mask: Optional attention mask.

        Returns:
            Output tensor [B, N_q, D_out].
        """
        B, N_q, _ = x.shape
        N_kv = context.shape[1]

        q = self.q_proj(x).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        if rope_q is not None:
            q = apply_rot_embed_cat(q, rope_q, half=self.rotate_half).type_as(v)
        if rope_k is not None:
            k = apply_rot_embed_cat(k, rope_k, half=self.rotate_half).type_as(v)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N_q, self.attn_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RSViTCrossBlock(nn.Module):
    """Transformer block with cross-attention and SwiGLU MLP.

    Used for Perceiver-style cross-attention where queries attend to a separate context.

    Args:
        dim: Dimension for queries.
        dim_context: Dimension for context. If None, same as dim.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        proj_drop: Projection dropout rate.
        init_values: Initial values for layer scale.
        drop_path: Stochastic depth rate.
        rotate_half: Use 'half' RoPE layout.
    """

    def __init__(
            self,
            dim: int,
            dim_context: Optional[int] = None,
            num_heads: int = 8,
            mlp_ratio: float = 4.,
            proj_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            rotate_half: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}

        self.norm1 = RmsNorm(dim, **dd)
        self.norm_context = RmsNorm(dim_context or dim, **dd)
        self.cross_attn = CrossAttentionRope(
            dim,
            dim_context=dim_context,
            num_heads=num_heads,
            qkv_bias=False,
            proj_bias=False,
            proj_drop=proj_drop,
            rotate_half=rotate_half,
            **dd,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RmsNorm(dim, **dd)
        self.mlp = SwiGLU(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            bias=False,
            drop=proj_drop,
            **dd,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            rope_q: Optional[torch.Tensor] = None,
            rope_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query input [B, N_q, D].
            context: Key/value context [B, N_kv, D_context].
            rope_q: RoPE embeddings for queries.
            rope_k: RoPE embeddings for keys/context.

        Returns:
            Output tensor [B, N_q, D].
        """
        x = x + self.drop_path1(
            self.ls1(
                self.cross_attn(
                    self.norm1(x),
                    self.norm_context(context),
                    rope_q=rope_q,
                    rope_k=rope_k,
                )
            )
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class RSViTBlock(nn.Module):
    """TRM Transformer block with pre-norm RMSNorm and SwiGLU MLP.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        num_prefix_tokens: Number of prefix tokens (e.g., cls token) excluded from RoPE.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        proj_drop: Projection dropout rate.
        init_values: Initial values for layer scale.
        drop_path: Stochastic depth rate.
        rotate_half: Use 'half' ROPE layout instead of default 'interleaved'.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            num_prefix_tokens: int = 0,
            mlp_ratio: float = 4.,
            proj_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            rotate_half: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}

        # Pre-norm with RMSNorm
        self.norm1 = RmsNorm(dim, **dd)
        self.attn = AttentionRope(
            dim,
            num_heads=num_heads,
            num_prefix_tokens=num_prefix_tokens,
            qkv_bias=False,
            proj_bias=False,
            proj_drop=proj_drop,
            rotate_half=rotate_half,
            **dd,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RmsNorm(dim, **dd)
        self.mlp = SwiGLU(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            bias=False,
            drop=proj_drop,
            **dd,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, N, D].
            rope: Rotary position embeddings (concatenated sin/cos).

        Returns:
            Output tensor [B, N, D].
        """
        # Pre-norm attention
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope=rope)))
        # Pre-norm MLP
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResidualMLP(nn.Module):
    """Residual MLP block with pre-norm.

    Args:
        dim: Input dimension.
        mlp_ratio: Ratio of mlp hidden dim to input dim.
        drop: Dropout rate.
        init_values: Initial values for layer scale.
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.,
            drop: float = 0.,
            init_values: Optional[float] = None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.norm = RmsNorm(dim, **dd)
        self.mlp = SwiGLU(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            bias=False,
            drop=drop,
            **dd,
        )
        self.ls = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ls(self.mlp(self.norm(x)))


class RSViT(nn.Module):
    """TRM Vision Transformer Encoder with recursive computation.

    Uses shared blocks (z_blocks for patches, y_block for class token) applied
    recursively. During training, outputs supervision signals at multiple steps.
    During inference, can halt early based on confidence.

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of input channels.
        num_classes: Number of classes for classification head.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        n_sup: Number of supervision steps (training).
        t_recursions: Number of recursions per supervision step.
        z_depth: Number of unique z_blocks (default: 1, backward compatible).
        recursion_mode: How blocks are called during t_recursions.
            - 'block': Each block repeated before moving to next, y_block once at end.
            - 'cycle': Cycle through all blocks, y_block after each full pass.
        rope_type: Type of rotary embeddings ('cat' or 'dinov3').
        rotate_half: Use 'half' RoPE layout instead of default 'interleaved'.
        z_init_mode: Mode for z_init ('single' broadcasts one token, 'per_patch' learns per position).
        y_block_mode: Mode for y_block ('self' for self-attention over [y,z], 'cross' for cross-attention y→z).
        drop_rate: Dropout rate.
        proj_drop_rate: Projection dropout rate.
        init_values: Initial values for layer scale.
        drop_path_rate: Stochastic depth rate.
        halt_threshold: Confidence threshold for early halting (inference).
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 768,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            n_sup: int = 4,
            t_recursions: int = 4,
            z_depth: int = 1,
            recursion_mode: str = 'block',
            rope_type: str = 'cat',
            rotate_half: bool = False,
            z_init_mode: str = 'single',
            y_block_mode: str = 'self',
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            init_values: Optional[float] = None,
            drop_path_rate: float = 0.,
            halt_threshold: float = 0.8,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}

        assert recursion_mode in ('block', 'cycle'), \
            f"recursion_mode must be 'block' or 'cycle', got '{recursion_mode}'"

        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_sup = n_sup
        self.t_recursions = t_recursions
        self.z_depth = z_depth
        self.recursion_mode = recursion_mode
        self.halt_threshold = halt_threshold

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True,
            **dd,
        )
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size
        self.norm_pre = nn.LayerNorm(embed_dim)

        # Learnable class token initialization
        self.y_init = nn.Parameter(torch.zeros(1, 1, embed_dim, **dd))

        # Learnable patch feature offset
        # 'single': one token broadcast to all patches (resolution independent)
        # 'per_patch': learned per position (requires interpolation for variable resolution)
        self.z_init_mode = z_init_mode
        if z_init_mode == 'single':
            self.z_init = nn.Parameter(torch.zeros(1, 1, embed_dim, **dd))
        else:
            self.z_init = nn.Parameter(torch.zeros(1, num_patches, embed_dim, **dd))

        self.drop_pre = nn.Dropout(p=drop_rate)

        # Rotary embeddings for spatial positions
        self.rope = create_rope_embed(
            rope_type=rope_type,
            dim=embed_dim,
            num_heads=num_heads,
            feat_shape=list(grid_size),
            rotate_half=rotate_half,
            **dd,
        )

        # Shared transformer blocks
        # z_blocks: process patch tokens only (no prefix tokens)
        self.z_blocks = nn.ModuleList([
            RSViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_prefix_tokens=0,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                init_values=init_values,
                drop_path=drop_path_rate,
                rotate_half=rotate_half,
                **dd,
            )
            for _ in range(z_depth)
        ])

        # y_block: class token attends to patches
        # 'self': self-attention over [y, z] (cls is prefix, excluded from RoPE)
        # 'cross': cross-attention where y queries z
        self.y_block_mode = y_block_mode
        if y_block_mode == 'cross':
            self.y_block = RSViTCrossBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                init_values=init_values,
                drop_path=drop_path_rate,
                rotate_half=rotate_half,
                **dd,
            )
        else:
            self.y_block = RSViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_prefix_tokens=1,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                init_values=init_values,
                drop_path=drop_path_rate,
                rotate_half=rotate_half,
                **dd,
            )

        # Final norm before classification
        self.norm = RmsNorm(embed_dim, **dd)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

        # Halting head (predicts if answer is correct)
        self.q_head = nn.Linear(embed_dim, 1, **dd)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        trunc_normal_(self.y_init, std=0.02)
        trunc_normal_(self.z_init, std=0.02)

        # Initialize linear layers
        self.apply(self._init_linear_weights)

    def _init_linear_weights(self, m: nn.Module):
        """Initialize linear layer weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Parameters to exclude from weight decay."""
        return {'y_init', 'z_init'}

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _run_y_block(
            self,
            y: torch.Tensor,
            z: torch.Tensor,
            rot_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Run y_block (class token attends to patches).

        Args:
            y: Class token [B, 1, D].
            z: Patch features [B, num_patches, D].
            rot_pos_embed: RoPE embeddings.

        Returns:
            Updated y tensor.
        """
        if self.y_block_mode == 'cross':
            return self.y_block(y, z, rope_k=rot_pos_embed)
        else:
            all_tokens = torch.cat([y, z], dim=1)
            all_tokens = self.y_block(all_tokens, rope=rot_pos_embed)
            y = all_tokens[:, :1, :]
            # Note: z unchanged in self-attention mode (we only extract y)
            return y

    def _run_supervision_step(
            self,
            z: torch.Tensor,
            y: torch.Tensor,
            rot_pos_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one supervision step (all z_block and y_block iterations).

        Args:
            z: Patch features [B, num_patches, D].
            y: Class token [B, 1, D].
            rot_pos_embed: RoPE embeddings.

        Returns:
            Updated (z, y) tuple.
        """
        if self.recursion_mode == 'cycle':
            # Cycle: y_block after each pass through all z_blocks
            for _ in range(self.t_recursions):
                for block in self.z_blocks:
                    z = block(z, rope=rot_pos_embed)
                y = self._run_y_block(y, z, rot_pos_embed)
        else:
            # Block: each z_block repeated, then y_block once
            for block in self.z_blocks:
                for _ in range(self.t_recursions):
                    z = block(z, rope=rot_pos_embed)
            y = self._run_y_block(y, z, rot_pos_embed)

        return z, y

    def forward_features(
            self,
            x: torch.Tensor,
            return_all_steps: bool = False,
    ) -> Union[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Forward pass through encoder.

        Args:
            x: Input tensor [B, C, H, W].
            return_all_steps: If True, return all supervision step outputs.

        Returns:
            If return_all_steps is False: class token features [B, D].
            If return_all_steps is True: dict with 'y_features' and 'z_features' lists.
        """
        B = x.shape[0]

        # Embed patches
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        x = self.norm_pre(x)

        # Add z_init offset (RoPE handles positional encoding)
        x = x + self.z_init
        z = self.drop_pre(x)  # [B, num_patches, embed_dim]

        # Initialize class token
        y = self.y_init.expand(B, -1, -1)  # [B, 1, embed_dim]

        # Get rotary embeddings (concatenated sin/cos)
        rot_pos_embed = self.rope.get_embed()

        if return_all_steps:
            y_features = []
            z_features = []

            for step in range(self.n_sup):
                z, y = self._run_supervision_step(z, y, rot_pos_embed)

                # Store features at each supervision step
                y_features.append(self.norm(y.squeeze(1)))
                z_features.append(z)

            return {'y_features': y_features, 'z_features': z_features}

        # Standard forward: run all supervision steps
        for _ in range(self.n_sup):
            z, y = self._run_supervision_step(z, y, rot_pos_embed)

        return self.norm(y.squeeze(1))

    def forward_head(
            self,
            x: torch.Tensor,
            pre_logits: bool = False,
            return_halt_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward through classification head.

        Args:
            x: Class token features [B, D].
            pre_logits: If True, return features before classification head.
            return_halt_logits: If True, also return halting logits.

        Returns:
            Classification logits [B, num_classes], optionally with halt logits.
        """
        if pre_logits:
            return x
        logits = self.head(x)
        halt_logits = self.q_head(x)
        if return_halt_logits:
            return logits, halt_logits
        # Add zero-scaled halt_logits to ensure gradient flow through q_head
        return logits + 0 * halt_logits.sum()

    def forward(
            self,
            x: torch.Tensor,
            return_all_steps: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].
            return_all_steps: If True, return all supervision step outputs for training.

        Returns:
            If return_all_steps is False: Classification logits [B, num_classes].
            If return_all_steps is True: Dict with:
                - 'step_logits': List of logits at each supervision step
                - 'halt_logits': List of halting logits at each supervision step
        """
        # Handle FX tracing: return_all_steps becomes a Proxy, use inference path
        if not isinstance(return_all_steps, bool):
            return_all_steps = False
        if return_all_steps:
            features = self.forward_features(x, return_all_steps=True)
            step_logits = []
            halt_logits = []
            for y_feat in features['y_features']:
                logits, halt = self.forward_head(y_feat, return_halt_logits=True)
                step_logits.append(logits)
                halt_logits.append(halt)
            return {'step_logits': step_logits, 'halt_logits': halt_logits}
        features = self.forward_features(x)
        return self.forward_head(features)

    def forward_with_halting(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with per-sample adaptive halting for inference.

        Halts each sample independently when its halting confidence exceeds the
        threshold. Returns final logits and the step at which each sample halted.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Tuple of (logits [B, num_classes], steps_taken [B]).
        """
        B = x.shape[0]
        device = x.device

        # Embed patches
        x = self.patch_embed(x)
        x = self.norm_pre(x)

        x = x + self.z_init
        z = self.drop_pre(x)

        y = self.y_init.expand(B, -1, -1)

        rot_pos_embed = self.rope.get_embed()

        # Track per-sample halting
        final_logits = torch.zeros(B, self.num_classes, device=device)
        steps_taken = torch.zeros(B, dtype=torch.long, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(self.n_sup):
            z, y = self._run_supervision_step(z, y, rot_pos_embed)

            # Check halting condition at each supervision step
            y_feat = self.norm(y.squeeze(1))
            logits, halt = self.forward_head(y_feat, return_halt_logits=True)

            # Per-sample halting decision
            halt_prob = torch.sigmoid(halt.squeeze(-1))  # [B]
            should_halt = (halt_prob > self.halt_threshold) & ~halted

            # Record final logits and step for newly halted samples
            final_logits[should_halt] = logits[should_halt]
            steps_taken[should_halt] = step + 1
            halted = halted | should_halt

            # Early exit if all samples have halted
            if halted.all():
                break

        # For samples that never halted, use final step output
        final_logits[~halted] = logits[~halted]
        steps_taken[~halted] = self.n_sup

        return final_logits, steps_taken


class RSPViT(nn.Module):
    """Recursive Supervision Perceiver ViT.

    Perceiver-style variant where latents (z) are a fixed-length sequence that
    cross-attends to input patches. Resolution independent.

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of input channels.
        num_classes: Number of classes for classification head.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_latents: Number of latent tokens (fixed, resolution independent).
        z_depth: Number of self-attention blocks for latents per recursion.
        latent_rope: Whether to use RoPE for latents (1D positional encoding).
        latent_feedback: Mode for feedback from y (cls token) to z (latents).
            - 'none': No feedback, z self-attends alone, y cross-attends to z (default).
            - 'joint': [y, z] joint self-attention, both updated together (like RSViT's y_block_mode='self').
            - 'asymmetric': z sees y during self-attention, but y only updated via cross-attention.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        n_sup: Number of supervision steps (training).
        t_recursions: Number of recursions per supervision step.
        rope_type: Type of rotary embeddings ('cat' or 'dinov3').
        rotate_half: Use 'half' RoPE layout instead of default 'interleaved'.
        drop_rate: Dropout rate.
        proj_drop_rate: Projection dropout rate.
        init_values: Initial values for layer scale.
        drop_path_rate: Stochastic depth rate.
        halt_threshold: Confidence threshold for early halting (inference).
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 768,
            num_heads: int = 12,
            num_latents: int = 64,
            z_depth: int = 1,
            latent_rope: bool = True,
            latent_feedback: str = 'none',
            mlp_ratio: float = 4.,
            n_sup: int = 4,
            t_recursions: int = 4,
            rope_type: str = 'dinov3',
            rotate_half: bool = False,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            init_values: Optional[float] = None,
            drop_path_rate: float = 0.,
            halt_threshold: float = 0.8,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}

        assert latent_feedback in ('none', 'joint', 'asymmetric'), \
            f"latent_feedback must be 'none', 'joint', or 'asymmetric', got '{latent_feedback}'"

        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.z_depth = z_depth
        self.latent_feedback = latent_feedback
        self.n_sup = n_sup
        self.t_recursions = t_recursions
        self.halt_threshold = halt_threshold

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True,
            **dd,
        )
        grid_size = self.patch_embed.grid_size
        self.norm_pre = nn.LayerNorm(embed_dim, **dd)
        self.drop_pre = nn.Dropout(p=drop_rate)

        # Learnable class token initialization
        self.y_init = nn.Parameter(torch.zeros(1, 1, embed_dim, **dd))

        # Learnable latent sequence (fixed length, resolution independent)
        self.z_init = nn.Parameter(torch.zeros(1, num_latents, embed_dim, **dd))

        # Rotary embeddings for patch spatial positions (2D grid)
        self.rope = create_rope_embed(
            rope_type=rope_type,
            dim=embed_dim,
            num_heads=num_heads,
            feat_shape=list(grid_size),
            rotate_half=rotate_half,
            **dd,
        )

        # Optional RoPE for latents (treated as pseudo-2D grid for axial RoPE)
        # Disabled if num_latents == 1 (single token has no meaningful position)
        self.latent_rope_enabled = latent_rope and num_latents > 1
        if self.latent_rope_enabled:
            # Factor num_latents into a 2D shape for axial RoPE
            # Prefer square shapes, otherwise use (num_latents, 1)
            sqrt_latents = int(num_latents ** 0.5)
            if sqrt_latents * sqrt_latents == num_latents:
                latent_grid = [sqrt_latents, sqrt_latents]
            else:
                latent_grid = [num_latents, 1]
            self.latent_rope = create_rope_embed(
                rope_type=rope_type,
                dim=embed_dim,
                num_heads=num_heads,
                feat_shape=latent_grid,
                rotate_half=rotate_half,
                **dd,
            )
        else:
            self.latent_rope = None

        # z_cross_block: latents cross-attend to patches (Perceiver-style)
        self.z_cross_block = RSViTCrossBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            proj_drop=proj_drop_rate,
            init_values=init_values,
            drop_path=drop_path_rate,
            rotate_half=rotate_half,
            **dd,
        )

        # z_self_blocks: latents self-attend (N blocks per recursion)
        # When latent_feedback is 'joint' or 'asymmetric', y is prepended to z,
        # so we use num_prefix_tokens=1 to exclude y from RoPE
        z_self_prefix_tokens = 1 if latent_feedback in ('joint', 'asymmetric') else 0
        self.z_self_blocks = nn.ModuleList([
            RSViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_prefix_tokens=z_self_prefix_tokens,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                init_values=init_values,
                drop_path=drop_path_rate,
                rotate_half=rotate_half,
                **dd,
            )
            for _ in range(z_depth)
        ])

        # y_block: class token cross-attends to latents
        # In 'joint' mode, y is already updated via joint self-attention, so y_block is optional
        # We keep it for potential extra refinement, but could be skipped
        self.y_block = RSViTCrossBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            proj_drop=proj_drop_rate,
            init_values=init_values,
            drop_path=drop_path_rate,
            rotate_half=rotate_half,
            **dd,
        )

        # Final norm before classification
        self.norm = RmsNorm(embed_dim, **dd)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

        # Halting head (predicts if answer is correct)
        self.q_head = nn.Linear(embed_dim, 1, **dd)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        trunc_normal_(self.y_init, std=0.02)
        trunc_normal_(self.z_init, std=0.02)

        # Initialize linear layers
        self.apply(self._init_linear_weights)

    def _init_linear_weights(self, m: nn.Module):
        """Initialize linear layer weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Parameters to exclude from weight decay."""
        return {'y_init', 'z_init'}

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _run_recursion_step(
            self,
            z: torch.Tensor,
            y: torch.Tensor,
            patches: torch.Tensor,
            rot_pos_embed: torch.Tensor,
            latent_rope_embed: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one recursion step (cross-attention + self-attention).

        Args:
            z: Latent states [B, num_latents, D].
            y: Class token [B, 1, D].
            patches: Embedded patches [B, num_patches, D].
            rot_pos_embed: RoPE embeddings for patches.
            latent_rope_embed: RoPE embeddings for latents (or None).

        Returns:
            Updated (z, y) tuple.
        """
        # Cross-attention: latents gather from patches
        z = self.z_cross_block(z, patches, rope_q=latent_rope_embed, rope_k=rot_pos_embed)

        # Self-attention with latent_feedback mode
        if self.latent_feedback == 'none':
            for self_block in self.z_self_blocks:
                z = self_block(z, rope=latent_rope_embed)
            y = self.y_block(y, z, rope_k=latent_rope_embed)

        elif self.latent_feedback == 'joint':
            combined = torch.cat([y, z], dim=1)
            for self_block in self.z_self_blocks:
                combined = self_block(combined, rope=latent_rope_embed)
            y = combined[:, :1, :]
            z = combined[:, 1:, :]

        else:  # 'asymmetric'
            combined = torch.cat([y, z], dim=1)
            for self_block in self.z_self_blocks:
                combined = self_block(combined, rope=latent_rope_embed)
            z = combined[:, 1:, :]
            y = self.y_block(y, z, rope_k=latent_rope_embed)

        return z, y

    def forward_features(
            self,
            x: torch.Tensor,
            return_all_steps: bool = False,
    ) -> Union[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Forward pass through encoder.

        Args:
            x: Input tensor [B, C, H, W].
            return_all_steps: If True, return all supervision step outputs.

        Returns:
            If return_all_steps is False: class token features [B, D].
            If return_all_steps is True: dict with 'y_features' and 'z_features' lists.
        """
        B = x.shape[0]

        # Embed patches
        patches = self.patch_embed(x)  # [B, num_patches, embed_dim]
        patches = self.norm_pre(patches)
        patches = self.drop_pre(patches)

        # Initialize latents and class token
        z = self.z_init.expand(B, -1, -1)  # [B, num_latents, embed_dim]
        y = self.y_init.expand(B, -1, -1)  # [B, 1, embed_dim]

        # Get rotary embeddings for patches (2D) and latents (1D, optional)
        rot_pos_embed = self.rope.get_embed()
        latent_rope_embed = self.latent_rope.get_embed() if self.latent_rope_enabled else None

        if return_all_steps:
            y_features = []
            z_features = []

            for step in range(self.n_sup):
                for t in range(self.t_recursions):
                    z, y = self._run_recursion_step(
                        z, y, patches, rot_pos_embed, latent_rope_embed
                    )

                # Store features at each supervision step
                y_features.append(self.norm(y.squeeze(1)))
                z_features.append(z)

            return {'y_features': y_features, 'z_features': z_features}
        else:
            # Standard forward: run all recursions
            total_recursions = self.n_sup * self.t_recursions

            for _ in range(total_recursions):
                z, y = self._run_recursion_step(
                    z, y, patches, rot_pos_embed, latent_rope_embed
                )

            return self.norm(y.squeeze(1))

    def forward_head(
            self,
            x: torch.Tensor,
            pre_logits: bool = False,
            return_halt_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward through classification head."""
        if pre_logits:
            return x
        logits = self.head(x)
        halt_logits = self.q_head(x)
        if return_halt_logits:
            return logits, halt_logits
        # Add zero-scaled halt_logits to ensure gradient flow through q_head
        return logits + 0 * halt_logits.sum()

    def forward(
            self,
            x: torch.Tensor,
            return_all_steps: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].
            return_all_steps: If True, return all supervision step outputs for training.

        Returns:
            If return_all_steps is False: Classification logits [B, num_classes].
            If return_all_steps is True: Dict with:
                - 'step_logits': List of logits at each supervision step
                - 'halt_logits': List of halting logits at each supervision step
        """
        # Handle FX tracing: return_all_steps becomes a Proxy, use inference path
        if not isinstance(return_all_steps, bool):
            return_all_steps = False
        if return_all_steps:
            features = self.forward_features(x, return_all_steps=True)
            step_logits = []
            halt_logits = []
            for y_feat in features['y_features']:
                logits, halt = self.forward_head(y_feat, return_halt_logits=True)
                step_logits.append(logits)
                halt_logits.append(halt)
            return {'step_logits': step_logits, 'halt_logits': halt_logits}
        features = self.forward_features(x)
        return self.forward_head(features)

    def forward_with_halting(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Forward pass with adaptive halting for inference."""
        B = x.shape[0]

        patches = self.patch_embed(x)
        patches = self.norm_pre(patches)
        patches = self.drop_pre(patches)

        z = self.z_init.expand(B, -1, -1)
        y = self.y_init.expand(B, -1, -1)

        rot_pos_embed = self.rope.get_embed()
        latent_rope_embed = self.latent_rope.get_embed() if self.latent_rope_enabled else None

        for step in range(self.n_sup):
            for t in range(self.t_recursions):
                # Cross-attention: latents gather from patches
                z = self.z_cross_block(z, patches, rope_q=latent_rope_embed, rope_k=rot_pos_embed)

                # Self-attention with latent_feedback mode
                if self.latent_feedback == 'none':
                    for self_block in self.z_self_blocks:
                        z = self_block(z, rope=latent_rope_embed)
                    y = self.y_block(y, z, rope_k=latent_rope_embed)

                elif self.latent_feedback == 'joint':
                    combined = torch.cat([y, z], dim=1)
                    for self_block in self.z_self_blocks:
                        combined = self_block(combined, rope=latent_rope_embed)
                    y = combined[:, :1, :]
                    z = combined[:, 1:, :]

                else:  # 'asymmetric'
                    combined = torch.cat([y, z], dim=1)
                    for self_block in self.z_self_blocks:
                        combined = self_block(combined, rope=latent_rope_embed)
                    z = combined[:, 1:, :]
                    y = self.y_block(y, z, rope_k=latent_rope_embed)

            y_feat = self.norm(y.squeeze(1))
            logits, halt = self.forward_head(y_feat, return_halt_logits=True)

            halt_prob = torch.sigmoid(halt)
            if halt_prob.mean() > self.halt_threshold:
                return logits, step + 1

        return logits, self.n_sup


class RSTViT(nn.Module):
    """Recursive Supervision Transformer Vision Transformer.

    A variant with fixed context patches, asymmetric updates, and Y-gated context
    injection. Key architectural differences from RSViT:
    - Context patches (x_ctx) computed once and never updated
    - Inner loop updates only z (via joint [y,z] self-attention, discarding y update)
    - Y updated once per supervision step (via joint [y,z] self-attention, discarding z update)
    - Y-gated context injection: y controls how original context flows into z
    - Per-sample halting in inference mode

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of input channels.
        num_classes: Number of classes for classification head.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        n_sup: Number of supervision steps (training).
        t_recursions: Number of recursions per supervision step.
        z_depth: Number of core_blocks for joint [y,z] self-attention.
        y_core_depth: Number of core_blocks to use for y update (default: 1).
        y_mlp_depth: Number of additional MLP layers for y update (default: 0).
        recursion_mode: How blocks are called during t_recursions.
            - 'block': Each block repeated before moving to next.
            - 'cycle': Cycle through all blocks, repeat the cycle.
        rope_type: Type of rotary embeddings ('cat' or 'dinov3').
        rotate_half: Use 'half' RoPE layout instead of default 'interleaved'.
        z_init_mode: How to initialize z ('single' broadcasts one token, 'per_patch' learns per position).
        ctx_gate_mode: Context gating mode ('uniform' for spatially uniform, 'spatial' for per-patch attention).
        drop_rate: Dropout rate.
        proj_drop_rate: Projection dropout rate.
        init_values: Initial values for layer scale.
        drop_path_rate: Stochastic depth rate.
        halt_threshold: Confidence threshold for early halting (inference).
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 768,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            n_sup: int = 4,
            t_recursions: int = 4,
            z_depth: int = 2,
            y_core_depth: int = 1,
            y_mlp_depth: int = 1,
            recursion_mode: str = 'block',
            rope_type: str = 'dinov3',
            rotate_half: bool = False,
            z_init_mode: str = 'single',
            ctx_gate_mode: str = 'uniform',
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            init_values: Optional[float] = None,
            drop_path_rate: float = 0.,
            halt_threshold: float = 0.8,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}

        assert ctx_gate_mode in ('uniform', 'spatial'), \
            f"ctx_gate_mode must be 'uniform' or 'spatial', got '{ctx_gate_mode}'"
        assert recursion_mode in ('block', 'cycle'), \
            f"recursion_mode must be 'block' or 'cycle', got '{recursion_mode}'"
        assert y_core_depth <= z_depth, \
            f"y_core_depth ({y_core_depth}) must be <= z_depth ({z_depth})"

        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_sup = n_sup
        self.t_recursions = t_recursions
        self.z_depth = z_depth
        self.y_core_depth = y_core_depth
        self.recursion_mode = recursion_mode
        self.halt_threshold = halt_threshold
        self.ctx_gate_mode = ctx_gate_mode

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True,
            **dd,
        )
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        # Pre-norm on context patches
        self.norm_pre = nn.LayerNorm(embed_dim, **dd)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Learnable class token initialization
        self.y_init = nn.Parameter(torch.zeros(1, 1, embed_dim, **dd))

        # Learnable z initialization
        # 'single': one token broadcast to all patches (resolution independent)
        # 'per_patch': learned per position (requires interpolation for variable resolution)
        self.z_init_mode = z_init_mode
        if z_init_mode == 'single':
            self.z_init = nn.Parameter(torch.zeros(1, 1, embed_dim, **dd))
        else:
            self.z_init = nn.Parameter(torch.zeros(1, num_patches, embed_dim, **dd))

        # RoPE for patch positions (prefix token excluded by core block)
        self.rope = create_rope_embed(
            rope_type=rope_type,
            dim=embed_dim,
            num_heads=num_heads,
            feat_shape=list(grid_size),
            rotate_half=rotate_half,
            **dd,
        )

        # Y-gated context injection into z
        # z <- z + ctx_scale * gate(y) * proj(x_ctx)
        self.ctx_proj = nn.Linear(embed_dim, embed_dim, bias=False, **dd)
        if ctx_gate_mode == 'uniform':
            # Uniform gating: same gate value broadcast across all patches
            self.ctx_gate = nn.Sequential(
                RmsNorm(embed_dim, **dd),
                nn.Linear(embed_dim, embed_dim, bias=True, **dd),
                nn.Sigmoid(),
            )
        else:
            # Spatial gating: per-patch gates via cross-attention scores
            # y queries x_ctx to get per-patch importance
            self.ctx_gate_q = nn.Linear(embed_dim, embed_dim, bias=False, **dd)
            self.ctx_gate_k = nn.Linear(embed_dim, embed_dim, bias=False, **dd)
        # Start small: allows gradient flow while not dominating early training
        self.ctx_scale = nn.Parameter(torch.tensor(0.1, **dd))

        # Core blocks for joint [y,z] self-attention
        # Prefix token count = 1 so RoPE is applied only to patch tokens
        self.core_blocks = nn.ModuleList([
            RSViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_prefix_tokens=1,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                init_values=init_values,
                drop_path=drop_path_rate,
                rotate_half=rotate_half,
                **dd,
            )
            for _ in range(z_depth)
        ])

        # Y-specific MLP depth for additional processing after attention
        self.y_mlps = nn.Sequential(*[
            ResidualMLP(embed_dim, mlp_ratio, proj_drop_rate, init_values=init_values, **dd)
            for _ in range(y_mlp_depth)
        ]) if y_mlp_depth > 0 else nn.Identity()

        # Final norm before heads
        self.norm = RmsNorm(embed_dim, **dd)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

        # Halting head
        self.q_head = nn.Linear(embed_dim, 1, **dd)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        trunc_normal_(self.y_init, std=0.02)
        trunc_normal_(self.z_init, std=0.02)
        self.apply(self._init_linear_weights)

    def _init_linear_weights(self, m: nn.Module):
        """Initialize linear layer weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Parameters to exclude from weight decay."""
        return {'y_init', 'z_init', 'ctx_scale'}

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _inject_context(
            self,
            z: torch.Tensor,
            x_ctx: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        """Inject context into z using y-gated mechanism.

        Args:
            z: Current latent state [B, num_patches, D].
            x_ctx: Fixed context patches [B, num_patches, D].
            y: Class token [B, 1, D].

        Returns:
            Updated z with gated context injection.
        """
        if self.ctx_gate_mode == 'uniform':
            # Uniform gating: gate(y) broadcast to all patches
            # z <- z + ctx_scale * gate(y) * proj(x_ctx)
            gate = self.ctx_gate(y)  # [B, 1, D]
            z = z + self.ctx_scale * gate * self.ctx_proj(x_ctx)
        else:
            # Spatial gating: per-patch attention weights
            # y queries x_ctx for per-patch relevance
            q = self.ctx_gate_q(y)  # [B, 1, D]
            k = self.ctx_gate_k(x_ctx)  # [B, N, D]
            N = x_ctx.shape[1]
            # Compute attention scores
            attn = (q @ k.transpose(-1, -2)) * (self.embed_dim ** -0.5)  # [B, 1, N]
            attn = torch.softmax(attn, dim=-1)  # [B, 1, N]
            # Scale by N so total injection magnitude matches uniform mode
            gate = N * attn.transpose(-1, -2)  # [B, N, 1]
            z = z + self.ctx_scale * gate * self.ctx_proj(x_ctx)
        return z

    def _update_z_only(
            self,
            y: torch.Tensor,
            z: torch.Tensor,
            rot_pos_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update only z via joint self-attention (y unchanged).

        Applies recursion_mode logic for block iteration.

        Args:
            y: Class token [B, 1, D] (unchanged).
            z: Latent state [B, num_patches, D].
            rot_pos_embed: RoPE embeddings.

        Returns:
            Tuple of (unchanged y, updated z).
        """
        if self.recursion_mode == 'cycle':
            # Cycle through all blocks, repeat t_recursions times
            for _ in range(self.t_recursions):
                for block in self.core_blocks:
                    tokens = torch.cat([y, z], dim=1)
                    tokens = block(tokens, rope=rot_pos_embed)
                    z = tokens[:, 1:, :]
        else:
            # Each block repeated t_recursions times before next
            for block in self.core_blocks:
                for _ in range(self.t_recursions):
                    tokens = torch.cat([y, z], dim=1)
                    tokens = block(tokens, rope=rot_pos_embed)
                    z = tokens[:, 1:, :]
        return y, z

    def _update_y_only(
            self,
            y: torch.Tensor,
            z: torch.Tensor,
            rot_pos_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update only y via attention + y_mlps (z unchanged).

        Uses first y_core_depth blocks with recursion_mode, then applies y_mlps.

        Args:
            y: Class token [B, 1, D].
            z: Latent state [B, num_patches, D] (unchanged).
            rot_pos_embed: RoPE embeddings.

        Returns:
            Tuple of (updated y, unchanged z).
        """
        if self.recursion_mode == 'cycle':
            for _ in range(self.t_recursions):
                for block in self.core_blocks[:self.y_core_depth]:
                    tokens = torch.cat([y, z], dim=1)
                    tokens = block(tokens, rope=rot_pos_embed)
                    y = tokens[:, :1, :]
        else:
            for block in self.core_blocks[:self.y_core_depth]:
                for _ in range(self.t_recursions):
                    tokens = torch.cat([y, z], dim=1)
                    tokens = block(tokens, rope=rot_pos_embed)
                    y = tokens[:, :1, :]
        y = self.y_mlps(y)
        return y, z

    def forward_features(
            self,
            x: torch.Tensor,
            return_all_steps: bool = False,
    ) -> Union[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Forward pass through encoder.

        Args:
            x: Input tensor [B, C, H, W].
            return_all_steps: If True, return all supervision step outputs.

        Returns:
            If return_all_steps is False: class token features [B, D].
            If return_all_steps is True: dict with 'y_features' and 'z_features' lists.
        """
        B = x.shape[0]

        # Fixed context patches (never updated)
        patches = self.patch_embed(x)  # [B, N, D]
        patches = self.norm_pre(patches)
        x_ctx = self.pos_drop(patches)

        # Initialize z from context + learned offset
        z = x_ctx + self.z_init  # broadcast OK if z_init is [1, 1, D]

        # Initialize class token
        y = self.y_init.expand(B, -1, -1)  # [B, 1, D]

        rot_pos_embed = self.rope.get_embed()

        if return_all_steps:
            y_features = []
            z_features = []

            for step in range(self.n_sup):
                # Context injection once per supervision step
                z = self._inject_context(z, x_ctx, y)

                # Update z (handles t_recursions internally)
                y, z = self._update_z_only(y, z, rot_pos_embed)

                # Update y once per supervision step
                y, z = self._update_y_only(y, z, rot_pos_embed)

                # Store features at each supervision step
                y_features.append(self.norm(y.squeeze(1)))
                z_features.append(z)

            return {'y_features': y_features, 'z_features': z_features}

        # Standard forward: run all supervision steps
        for _ in range(self.n_sup):
            # Context injection once per supervision step
            z = self._inject_context(z, x_ctx, y)
            # Update z (handles t_recursions internally)
            y, z = self._update_z_only(y, z, rot_pos_embed)
            # Update y once
            y, z = self._update_y_only(y, z, rot_pos_embed)

        return self.norm(y.squeeze(1))

    def forward_head(
            self,
            x: torch.Tensor,
            pre_logits: bool = False,
            return_halt_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward through classification head.

        Args:
            x: Class token features [B, D].
            pre_logits: If True, return features before classification head.
            return_halt_logits: If True, also return halting logits.

        Returns:
            Classification logits [B, num_classes], optionally with halt logits.
        """
        if pre_logits:
            return x
        logits = self.head(x)
        halt_logits = self.q_head(x)
        if return_halt_logits:
            return logits, halt_logits
        # Add zero-scaled halt_logits to ensure gradient flow through q_head
        return logits + 0 * halt_logits.sum()

    def forward(
            self,
            x: torch.Tensor,
            return_all_steps: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].
            return_all_steps: If True, return all supervision step outputs for training.

        Returns:
            If return_all_steps is False: Classification logits [B, num_classes].
            If return_all_steps is True: Dict with:
                - 'step_logits': List of logits at each supervision step
                - 'halt_logits': List of halting logits at each supervision step
        """
        # Handle FX tracing: return_all_steps becomes a Proxy, use inference path
        if not isinstance(return_all_steps, bool):
            return_all_steps = False
        if return_all_steps:
            features = self.forward_features(x, return_all_steps=True)
            step_logits = []
            halt_logits = []
            for y_feat in features['y_features']:
                logits, halt = self.forward_head(y_feat, return_halt_logits=True)
                step_logits.append(logits)
                halt_logits.append(halt)
            return {'step_logits': step_logits, 'halt_logits': halt_logits}
        features = self.forward_features(x)
        return self.forward_head(features)

    def forward_with_halting(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with per-sample adaptive halting for inference.

        Halts each sample independently when its halting confidence exceeds the
        threshold. Returns final logits and the step at which each sample halted.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Tuple of (logits [B, num_classes], steps_taken [B]).
        """
        B = x.shape[0]
        device = x.device

        # Fixed context patches
        patches = self.patch_embed(x)
        patches = self.norm_pre(patches)
        x_ctx = self.pos_drop(patches)

        # Initialize z from context + learned offset
        z = x_ctx + self.z_init
        y = self.y_init.expand(B, -1, -1)

        rot_pos_embed = self.rope.get_embed()

        # Track per-sample halting
        final_logits = torch.zeros(B, self.num_classes, device=device)
        steps_taken = torch.zeros(B, dtype=torch.long, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(self.n_sup):
            # Context injection once per supervision step
            z = self._inject_context(z, x_ctx, y)

            # Update z (handles t_recursions internally)
            y, z = self._update_z_only(y, z, rot_pos_embed)

            # Update y once
            y, z = self._update_y_only(y, z, rot_pos_embed)

            # Check halting condition
            y_feat = self.norm(y.squeeze(1))
            logits, halt = self.forward_head(y_feat, return_halt_logits=True)

            # Per-sample halting decision
            halt_prob = torch.sigmoid(halt.squeeze(-1))  # [B]
            should_halt = (halt_prob > self.halt_threshold) & ~halted

            # Record final logits and step for newly halted samples
            final_logits[should_halt] = logits[should_halt]
            steps_taken[should_halt] = step + 1
            halted = halted | should_halt

            # Early exit if all samples have halted
            if halted.all():
                break

        # For samples that never halted, use final step output
        final_logits[~halted] = logits[~halted]
        steps_taken[~halted] = self.n_sup

        return final_logits, steps_taken


def _cfg(url: str = '', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'rsvit_tiny_patch16_224.untrained': _cfg(),
    'rsvit_small_patch16_224.untrained': _cfg(),
    'rsvit_base_patch16_224.untrained': _cfg(),
    'rspvit_tiny_patch16_224.untrained': _cfg(),
    'rspvit_small_patch16_224.untrained': _cfg(),
    'rspvit_base_patch16_224.untrained': _cfg(),
    'rstvit_tiny_patch16_224.untrained': _cfg(),
    'rstvit_small_patch16_224.untrained': _cfg(),
    'rstvit_base_patch16_224.untrained': _cfg(),
})


def _create_rsvit(variant: str, pretrained: bool = False, **kwargs) -> RSViT:
    """Create RSViT model."""
    return build_model_with_cfg(
        RSViT,
        variant,
        pretrained,
        **kwargs,
    )


@register_model
def rsvit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> RSViT:
    """RSViT-Tiny with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=192, num_heads=3)
    return _create_rsvit('rsvit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rsvit_small_patch16_224(pretrained: bool = False, **kwargs) -> RSViT:
    """RSViT-Small with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=384, num_heads=6)
    return _create_rsvit('rsvit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rsvit_base_patch16_224(pretrained: bool = False, **kwargs) -> RSViT:
    """RSViT-Base with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=768, num_heads=12)
    return _create_rsvit('rsvit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


def _create_rspvit(variant: str, pretrained: bool = False, **kwargs) -> RSPViT:
    """Create RSPViT model."""
    return build_model_with_cfg(
        RSPViT,
        variant,
        pretrained,
        **kwargs,
    )


@register_model
def rspvit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> RSPViT:
    """RSPViT-Tiny (Perceiver-style) with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=192, num_heads=3, num_latents=64)
    return _create_rspvit('rspvit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rspvit_small_patch16_224(pretrained: bool = False, **kwargs) -> RSPViT:
    """RSPViT-Small (Perceiver-style) with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=384, num_heads=6, num_latents=64)
    return _create_rspvit('rspvit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rspvit_base_patch16_224(pretrained: bool = False, **kwargs) -> RSPViT:
    """RSPViT-Base (Perceiver-style) with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=768, num_heads=12, num_latents=64)
    return _create_rspvit('rspvit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


def _create_rstvit(variant: str, pretrained: bool = False, **kwargs) -> RSTViT:
    """Create RSTViT model."""
    return build_model_with_cfg(
        RSTViT,
        variant,
        pretrained,
        **kwargs,
    )


@register_model
def rstvit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> RSTViT:
    """RSTViT-Tiny with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=192, num_heads=3)
    return _create_rstvit('rstvit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rstvit_small_patch16_224(pretrained: bool = False, **kwargs) -> RSTViT:
    """RSTViT-Small with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=384, num_heads=6)
    return _create_rstvit('rstvit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rstvit_base_patch16_224(pretrained: bool = False, **kwargs) -> RSTViT:
    """RSTViT-Base with patch size 16 and image size 224."""
    model_args = dict(patch_size=16, embed_dim=768, num_heads=12)
    return _create_rstvit('rstvit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
