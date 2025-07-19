""" NaFlex Vision Transformer

An improved version of the Vision Transformer with:
1. Encapsulated embedding and position encoding in a single module
2. Support for linear patch embedding on pre-patchified inputs
3. Support for NaFlex variable aspect, variable resolution
4. Support for FlexiViT variable patch size
5. Support for NaViT fractional/factorized position embedding

Based on ideas from:
- Original Vision Transformer: https://arxiv.org/abs/2010.11929
- FlexiViT: https://arxiv.org/abs/2212.08013
- NaViT: https://arxiv.org/abs/2307.06304
- NaFlex (SigLip-2): https://arxiv.org/abs/2502.14786

Hacked together by / Copyright 2025, Ross Wightman, Hugging Face
"""

import logging
import math
from dataclasses import dataclass, fields, replace
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import (
    AttentionPoolLatent,
    Mlp,
    LayerNorm,
    PatchDropoutWithIndices,
    PatchEmbedInterpolator,
    _assert,
    to_2tuple,
    get_act_layer,
    get_norm_layer,
    apply_keep_indices_nlc,
    disable_compiler,
)
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_function, register_notrace_module
from ._manipulate import checkpoint, named_apply
from ._registry import register_model, generate_default_cfgs
from .eva import EvaBlock
from .vision_transformer import Block, global_pool_nlc

__all__ = ['NaFlexVitCfg', 'NaFlexVit']


_logger = logging.getLogger(__name__)


@dataclass
class NaFlexVitCfg:
    """Configuration for FlexVit model.

    This dataclass contains the bulk of model configuration parameters,
    with core parameters (img_size, in_chans, num_classes, etc.) remaining
    as direct constructor arguments for API compatibility.
    """
    # Architecture parameters
    patch_size: Union[int, Tuple[int, int]] = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    scale_mlp_norm: bool = False  # Apply scaling norm to MLP

    # Attention parameters
    qkv_bias: bool = True
    qk_norm: bool = False
    proj_bias: bool = True
    attn_drop_rate: float = 0.0
    scale_attn_inner_norm: bool = False  # Apply scaling norm to attn context

    # Regularization
    init_values: Optional[float] = None  # Layer-scale init values (layer-scale enabled if not None)
    drop_rate: float = 0.0  # Dropout rate for classifier
    pos_drop_rate: float = 0.0  # Dropout rate for position embeddings
    patch_drop_rate: float = 0.0  # Dropout rate for patch tokens
    proj_drop_rate: float = 0.0  # Dropout rate for linear projections
    drop_path_rate: float = 0.0  # Stochastic depth drop rate

    # Prefix token configuration
    class_token: bool = False  # Use class token
    reg_tokens: int = 0  # Number of register tokens

    # Position embedding configuration
    pos_embed: str = 'learned'  # Type of position embedding ('learned', 'factorized', 'rope', 'none')
    pos_embed_grid_size: Optional[Tuple[int, int]] = (16, 16)  # Grid size for position embedding initialization
    pos_embed_interp_mode: str = 'bicubic'  # Interpolation mode for position embedding resizing
    pos_embed_ar_preserving: bool = False  # Whether to preserve aspect ratio during position embedding interpolation
    pos_embed_use_grid_sample: bool = False  # Whether to use grid_sample for naflex position embedding interpolation

    # ROPE specific configuration
    rope_type: str = ''  # ROPE type: '' or 'none' for no ROPE, 'axial' for standard, 'mixed' for learnable frequencies
    rope_temperature: float = 10000.0  # Temperature for ROPE frequency computation
    rope_ref_feat_shape: Optional[Tuple[int, int]] = None
    rope_grid_offset: float = 0.  # Grid offset for non-pixel ROPE mode
    rope_grid_indexing: str = 'ij'  # Grid indexing mode for ROPE ('ij' or 'xy')

    # Image processing
    dynamic_img_pad: bool = False  # Whether to enable dynamic padding for variable resolution

    # Other architecture choices
    pre_norm: bool = False  # Whether to apply normalization before attention/MLP layers (start of blocks)
    final_norm: bool = True  # Whether to apply final normalization before pooling and classifier (end of blocks)
    fc_norm: Optional[bool] = None  # Whether to normalize features before final classifier (after pooling)

    # Global pooling setup
    global_pool: str = 'map'  # Type of global pooling for final sequence
    pool_include_prefix: bool = False  # Whether to include class/register prefix tokens in global pooling
    attn_pool_num_heads: Optional[int] = None  # Override num_heads for attention pool
    attn_pool_mlp_ratio: Optional[float] = None   # Override mlp_ratio for attention pool

    # Weight initialization
    weight_init: str = ''  # Weight initialization scheme
    fix_init: bool = True  # Apply weight initialization fix (scaling w/ layer index)

    # Embedding configuration
    embed_proj_type: str = 'linear'  # Type of embedding layer ('conv' or 'linear')
    input_norm_layer: Optional[str] = None  # Normalization layer for embeddings input (before input projection)
    embed_norm_layer: Optional[str] = None  # Normalization layer for embeddings (after input projection)

    # Layer implementations
    norm_layer: Optional[str] = None  # Normalization layer for transformer blocks
    act_layer: Optional[str] = None  # Activation layer for MLP blocks
    block_fn: Optional[str] = None  # Transformer block implementation class name
    mlp_layer: Optional[str] = None  # MLP implementation class name

    # EVA-specific parameters
    attn_type: str = 'standard'  # Attention type: 'standard', 'eva', 'rope'
    swiglu_mlp: bool = False  # Use SwiGLU MLP variant
    qkv_fused: bool = True  # Whether to use fused QKV projections

    # Variable patch size support
    enable_patch_interpolator: bool = False  # Enable dynamic patch size support


def _overlay_kwargs(cfg: NaFlexVitCfg, **kwargs) -> NaFlexVitCfg:
    """Overlay kwargs onto config, replacing config values with provided kwargs."""
    # Only update fields that exist in the config
    config_fields = set(cfg.__dataclass_fields__.keys())
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

    if config_kwargs:
        cfg = replace(cfg, **config_kwargs)

    return cfg


def batch_patchify(
        x: torch.Tensor,
        patch_size: Tuple[int, int],
        pad: bool = True,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Patchify a batch of images.

    Args:
        x: Input tensor of shape [B, C, H, W].
        patch_size: Patch dimensions (patch_h, patch_w).
        pad: Whether to pad images to be divisible by patch size.

    Returns:
        Tuple of (patches, grid_size) where patches has shape [B, N, P*P*C]
        and grid_size is (num_patches_h, num_patches_w).
    """
    B, C, H, W = x.shape
    ph, pw = patch_size

    # Ensure the image is divisible by patch size
    if pad and (H % ph != 0 or W % pw != 0):
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        x = F.pad(x, (0, pad_w, 0, pad_h))

    nh, nw = H // ph, W // pw
    patches = x.view(B, C, nh, ph, nw, pw).permute(0, 2, 4, 3, 5, 1).reshape(B, nh * nw, ph * pw * C)
    # FIXME confirm we want 'channels last' in the patch channel layout, egg ph, ph, C instead of C, ph, hw

    return patches, (nh, nw)


def calculate_naflex_grid_sizes(_coord: torch.Tensor):
    # Calculate the appropriate grid size from coords
    max_y = _coord[:, :, 0].amax(dim=1) + 1
    max_x = _coord[:, :, 1].amax(dim=1) + 1
    return [(int(h.item()), int(w.item())) for h, w in zip(max_y, max_x)]


class NaFlexRopeIterator:
    """Iterator for generating batched ROPE embeddings for mixed mode with multiple grid sizes."""

    def __init__(
        self,
        rope_module,
        size_to_indices: Dict[Tuple[int, int], List[int]],
        unique_sizes: List[Tuple[int, int]],
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.rope = rope_module
        self.size_to_indices = size_to_indices
        self.unique_sizes = unique_sizes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dtype = dtype
        self.device = device
        self.depth = rope_module.depth
        self.num_heads = rope_module.num_heads
        self.head_dim = 2 * rope_module.dim // rope_module.num_heads
        self._depth_idx = 0

        # Pre-compute embeddings for each unique size
        self._embeddings_per_size = {}
        for grid_size in unique_sizes:
            # get_embed returns all depths at once for mixed mode
            rope_embed = rope_module.get_embed(shape=grid_size)
            self._embeddings_per_size[grid_size] = rope_embed

    def __iter__(self):
        self._depth_idx = 0
        return self

    @disable_compiler
    def __next__(self):
        if self._depth_idx >= self.depth:
            raise StopIteration

        # Create batch tensor for current depth
        batch_embed = torch.zeros(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            dtype=self.dtype, device=self.device
        )

        # Fill in embeddings for each unique grid size
        for grid_size in self.unique_sizes:
            h, w = grid_size
            actual_len = h * w
            batch_indices = self.size_to_indices[grid_size]

            # Get pre-computed embeddings for this size at current depth
            embed = self._embeddings_per_size[grid_size][self._depth_idx]  # [num_heads, H*W, dim]

            # Assign to batch indices
            for bi in batch_indices:
                batch_embed[bi, :, :actual_len, :] = embed[:, :actual_len, :]

        self._depth_idx += 1
        return batch_embed


def get_block_fn(cfg: NaFlexVitCfg) -> Callable:
    """Get appropriate block function based on configuration.

    Returns a partially applied block constructor with EVA-specific
    or conflicting parameters pre-configured if needed.
    """
    # Check if we need EVA block features
    use_eva_features = (
        cfg.attn_type in ('eva', 'rope') or
        cfg.rope_type not in ('', 'none') or  # Any ROPE type requires EVA blocks
        cfg.swiglu_mlp
    )

    if use_eva_features:
        # Determine attention type based on rope_type if not explicitly set
        attn_type = cfg.attn_type
        if attn_type == 'standard' and cfg.rope_type not in ('', 'none'):
            attn_type = 'rope'

        num_prefix_tokens = (1 if cfg.class_token else 0) + cfg.reg_tokens
        return partial(
            EvaBlock,
            attn_type=attn_type,
            swiglu_mlp=cfg.swiglu_mlp,
            scale_mlp=cfg.scale_mlp_norm,
            scale_attn_inner=cfg.scale_attn_inner_norm,
            qkv_fused=cfg.qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
        )
    else:
        # Standard ViT block
        block_fn = cfg.block_fn or Block
        if cfg.scale_mlp_norm or cfg.scale_attn_inner_norm:
            # param names differ between EVA vs non-EVA block types
            block_fn = partial(
                block_fn,
                scale_mlp_norm=cfg.scale_mlp_norm,
                scale_attn_norm=cfg.scale_attn_inner_norm
            )
        return block_fn


@register_notrace_module
class NaFlexEmbeds(nn.Module):
    """NaFlex Embedding module for Vision Transformers.

    This module encapsulates the complete embedding process for Vision Transformers,
    supporting both standard and NaFlex (NaViT + FlexiViT) functionality:

    1. Patch embedding (via Conv2d or Linear)
    2. Class and register token preparation
    3. Position embedding addition with interpolation support
    4. Pre-normalization (if requested)
    5. Dropout application

    NaFlex capabilities include:
    - Variable aspect ratio and resolution via patch coordinates
    - Patch type indicators for handling padding tokens in attention
    - Flexible position embedding interpolation for arbitrary grid sizes
    - Support for factorized position embeddings

    The patch embedding can be one of two types:
    - Conv2d-based (default): For standard image inputs [B, C, H, W]
    - Linear-based: For pre-patchified inputs [B, N, P*P*C]

    Args:
        patch_size: Size of patches for patch embedding
        in_chans: Number of input image channels
        embed_dim: Dimensionality of patch embedding
        proj_type: Type of embedding projection layer ('conv' or 'linear')
        input_norm_layer: Normalization layer applied to input (linear mode only)
        proj_norm_layer: Normalization layer applied after projection
        pos_embed: Type of position embedding ('learned', 'factorized', 'none')
        pos_drop_rate: Dropout rate for position embeddings
        class_token: Whether to include a class token
        reg_tokens: Number of register tokens to include
        bias: Whether to use bias in projection layers
        dynamic_img_pad: Whether to enable dynamic padding for variable resolution
        pos_embed_grid_size: Grid size for position embedding initialization
        pos_embed_interp_mode: Interpolation mode for position embedding resizing
        pos_embed_ar_preserving: Whether to preserve aspect ratio during position embedding interpolation
        default_img_size: Default image size for position embedding grid calculation
    """

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            proj_type: Optional[str] = None,
            proj_bias: bool = True,
            class_token: bool = True,
            reg_tokens: int = 0,
            dynamic_img_pad: bool = False,
            default_img_size: Optional[Union[int, Tuple[int, int]]] = None,
            pos_embed: str = 'learned',
            pos_embed_grid_size: Optional[Tuple[int, int]] = (14, 14),
            pos_embed_interp_mode: str = 'bicubic',
            pos_embed_ar_preserving: bool = False,
            pos_embed_use_grid_sample: bool = False,
            input_norm_layer: Optional[Type[nn.Module]] = None,
            proj_norm_layer: Union[bool, Optional[Type[nn.Module]]] = None,
            norm_layer: Optional[Type[nn.Module]] = None,
            pos_drop_rate: float = 0.,
            enable_patch_interpolator: bool = False,
    ) -> None:
        """Initialize NaFlexEmbeds module.

        Args:
            patch_size: Size of patches for patch embedding.
            in_chans: Number of input image channels.
            embed_dim: Dimensionality of patch embedding.
            proj_type: Type of embedding projection layer ('conv' or 'linear').
            proj_bias: Whether to use bias in projection layers.
            class_token: Whether to include a class token.
            reg_tokens: Number of register tokens to include.
            dynamic_img_pad: Whether to enable dynamic padding for variable resolution.
            default_img_size: Default image size for position embedding grid calculation.
            pos_embed: Type of position embedding ('learned', 'factorized', 'none').
            pos_embed_grid_size: Grid size for position embedding initialization.
            pos_embed_interp_mode: Interpolation mode for position embedding resizing.
            pos_embed_ar_preserving: Whether to preserve aspect ratio during interpolation.
            input_norm_layer: Normalization layer applied to input (linear mode only).
            proj_norm_layer: Normalization layer applied after projection.
            norm_layer: Default normalization layer.
            pos_drop_rate: Dropout rate for position embeddings.
            enable_patch_interpolator: Enable dynamic patch size support.
        """
        super().__init__()
        self.has_class_token = class_token
        self.num_reg_tokens = reg_tokens
        self.pos_embed_interp_mode = pos_embed_interp_mode
        self.pos_embed_ar_preserving = pos_embed_ar_preserving
        self.pos_embed_use_grid_sample = pos_embed_use_grid_sample
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.dynamic_img_pad = dynamic_img_pad
        self.enable_patch_interpolator = enable_patch_interpolator

        # Calculate number of prefix tokens
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens

        # Create class and register tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None

        # Calculate grid size and number of patches
        self.default_img_size: Optional[Tuple[int, int]] = None
        self.pos_embed_grid_size: Optional[Tuple[int, int]] = None  # Grid size used for learned pos embed init
        if pos_embed_grid_size is not None:
            # Highest priority, use provided pos_embed_grid_size
            self.pos_embed_grid_size = pos_embed_grid_size
        elif default_img_size is not None:
            # Fallback to calculating grid size from img_size + patch_size if img size provided.
            self.default_img_size = to_2tuple(default_img_size)
            self.pos_embed_grid_size = tuple([s // p for s, p in zip(self.default_img_size, self.patch_size)])

        # Determine patch embedding type (linear or conv2d)
        if proj_type == 'linear':
            # Create linear projection for pre-patchified inputs
            # Input dimension is patch_size^2 * in_chans
            patch_dim = self.patch_size[0] * self.patch_size[1] * in_chans
            assert not (input_norm_layer is True and norm_layer is None), \
                "`norm_layer` must be given when input_norm_layer=True"
            input_norm_layer = norm_layer if input_norm_layer is True else (input_norm_layer or None)
            self.norm_input = input_norm_layer(patch_dim) if input_norm_layer else None
            self.proj = nn.Linear(patch_dim, embed_dim, bias=proj_bias)
            self.flatten = False
            self.is_linear = True
        else:
            # Default to convolutional patch embedding for image inputs
            assert not input_norm_layer
            self.norm_input = None
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=proj_bias
            )
            self.flatten = True
            self.is_linear = False

        # Create patch embedding interpolator if enabled
        if self.enable_patch_interpolator:
            self.patch_interpolator = PatchEmbedInterpolator(
                base_patch_size=self.patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                interpolation=pos_embed_interp_mode,
                antialias=True,
            )
        else:
            self.patch_interpolator = None

        # Create normalization layer after the projection
        assert not (proj_norm_layer is True and norm_layer is None), \
            "`norm_layer` must be given when proj_norm_layer=True"
        proj_norm_layer = norm_layer if proj_norm_layer is True else (proj_norm_layer or None)
        self.norm = proj_norm_layer(embed_dim) if proj_norm_layer else nn.Identity()

        # Create position embedding if needed - only for patches, never for prefix tokens
        if pos_embed in ('factorized', 'learned') and self.pos_embed_grid_size is None:
            raise ValueError(
                "Cannot initialize position embeddings without grid_size."
                "Please provide img_size or pos_embed_grid_size.")
        self.pos_embed: Optional[torch.Tensor] = None
        self.pos_embed_y: Optional[torch.Tensor] = None
        self.pos_embed_x: Optional[torch.Tensor] = None
        if not pos_embed or pos_embed == 'none':
            self.pos_embed_type = 'none'
        elif pos_embed == 'factorized':
            assert self.pos_embed_grid_size is not None
            h, w = self.pos_embed_grid_size
            self.pos_embed_type = 'factorized'
            self.pos_embed_y = nn.Parameter(torch.randn(1, h, embed_dim) * .02)
            self.pos_embed_x = nn.Parameter(torch.randn(1, w, embed_dim) * .02)
        else:
            assert self.pos_embed_grid_size is not None
            h, w = self.pos_embed_grid_size
            self.pos_embed = nn.Parameter(torch.randn(1, h, w, embed_dim) * .02)
            self.pos_embed_type = 'learned'

        # Dropout layer
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

    def feature_info(self, location) -> Dict[str, Any]:
        """Get feature information for feature extraction.

        Args:
            location: Feature extraction location identifier

        Returns:
            Dictionary containing feature channel count and reduction factor
        """
        return dict(num_chs=self.embed_dim, reduction=self.patch_size)

    def feat_ratio(self, as_scalar: bool = True) -> Union[int, Tuple[int, int]]:
        """Get the feature reduction ratio (stride) of the patch embedding.

        Args:
            as_scalar: Whether to return the maximum dimension as a scalar

        Returns:
            Feature reduction ratio as scalar or tuple
        """
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate grid (feature) size for given image size.

        Takes into account dynamic padding when enabled.

        Args:
            img_size: Input image size as (height, width)

        Returns:
            Grid size as (grid_height, grid_width)
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    @disable_compiler
    def _apply_learned_naflex_pos_embed(
            self,
            x: torch.Tensor,
            patch_coord: torch.Tensor,
    ) -> None:
        """Apply learned position embeddings to NaFlex batch in-place.

        Interpolates learned 2D position embeddings for each sample in the batch
        based on their individual grid sizes.

        Args:
            x: Input tensor to add position embeddings to [B, N, C]
            patch_coord: Patch coordinates [B, N, 2] with (y, x) values
        """
        # Calculate grid sizes from patch coordinates
        naflex_grid_sizes = calculate_naflex_grid_sizes(patch_coord)
        orig_h, orig_w = self.pos_embed.shape[1:3]
        pos_embed_nchw = self.pos_embed.permute(0, 3, 1, 2).float()  # B,C,H,W

        def _interp2d(size):
            """
            Return a flattened positional-embedding grid at an arbitrary spatial resolution.

            Converts the learned 2-D table stored in NCHW format (pos_embed_nchw) into
            a (1, H*W, C) sequence that matches the requested size.
            """
            if (size[0] == orig_h) and (size[1] == orig_w):
                pos_embed_flat = self.pos_embed.reshape(1, orig_h * orig_w, -1)
            else:
                _interp_size = to_2tuple(max(size)) if self.pos_embed_ar_preserving else size
                pos_embed_flat = F.interpolate(
                    pos_embed_nchw,
                    size=_interp_size,
                    mode=self.pos_embed_interp_mode,
                    align_corners=False,
                    antialias=True,
                )[:, :, :size[0], :size[1]].flatten(2).transpose(1, 2)
            return pos_embed_flat.to(dtype=x.dtype)

        # Determine unique grid sizes to avoid duplicate interpolation
        size_to_indices: Dict[Tuple[int, int], List[int]] = {}
        for bi, k in enumerate(naflex_grid_sizes):
            # k = h << 16 | w  # FIXME can get jit compat with this
            size_to_indices.setdefault(k, []).append(bi)

        for k, batch_indices in size_to_indices.items():
            # h, w = k >> 16, k & 0xFFFF  # FIXME can get jit compat with this
            # Interpolate only once for this (h, w)
            pos_embed_flat = _interp2d(k)
            seq_len = min(x.shape[1], pos_embed_flat.shape[1])
            x[:, :seq_len].index_add_(
                0,
                torch.as_tensor(batch_indices, device=x.device),
                pos_embed_flat[:, :seq_len].expand(len(batch_indices), -1, -1)
            )

    @disable_compiler
    def _apply_learned_naflex_pos_embed_grid_sample(
            self,
            x: torch.Tensor,
            patch_coord: torch.Tensor,
    ) -> None:
        """Apply learned position embeddings to NaFlex batch using grid_sample.

        Uses F.grid_sample for efficient interpolation of learned 2D position embeddings
        based on patch coordinates. Based on proposal by https://github.com/stas-sl

        Args:
            x: Input tensor to add position embeddings to [B, N, C]
            patch_coord: Patch coordinates [B, N, 2] with (y, x) values
        """
        device = x.device
        B, N, C = x.shape
        shapes = patch_coord.max(dim=1).values + 1  # (B, 2) containing [h_i, w_i]

        if self.pos_embed_ar_preserving:
            L_i = shapes.amax(dim=1)  # (B,) max(h_i, w_i)
            L_global = L_i.amax()
            grid_size_y = grid_size_x = L_global
            scale_x = scale_y = L_global / L_i  # uniform zoom (B,)
        else:
            grid_size_y, grid_size_x = shapes.amax(dim=0)  # (2,)
            scale_y = grid_size_y / shapes[:, 0]  # vertical zoom (B,)
            scale_x = grid_size_x / shapes[:, 1]  # horizontal zoom (B,)

        theta = torch.zeros(B, 2, 3, device=device, dtype=torch.float32)
        theta[:, 0, 0] = scale_x
        theta[:, 1, 1] = scale_y
        theta[:, 0, 2] = scale_x - 1  # translate x
        theta[:, 1, 2] = scale_y - 1  # translate y

        grid = F.affine_grid(theta, (B, C, grid_size_y, grid_size_x), align_corners=False)
        pos_embed = F.grid_sample(
            self.pos_embed.permute(0, 3, 1, 2).expand(B, -1, -1, -1).float(),
            grid,
            mode=self.pos_embed_interp_mode,
            align_corners=False,
            padding_mode='border',
        ).to(dtype=x.dtype)  # (B, C, H_out, W_out)

        bi = torch.arange(B, device=device).unsqueeze(1)
        x += pos_embed[bi, :, patch_coord[..., 0], patch_coord[..., 1]]  # NOTE leave as '+='

    def _apply_learned_pos_embed(
            self,
            x: torch.Tensor,
            grid_size: List[int],
    ) -> None:
        """Apply learned position embeddings to standard 2D batch in-place.

        Interpolates learned 2D position embeddings to match the specified grid size.

        Args:
            x: Input tensor to add position embeddings to [B, H*W, C]
            grid_size: Target grid size as [height, width]
        """
        orig_h, orig_w = self.pos_embed.shape[1:3]
        if grid_size[0] == orig_h and grid_size[1] == orig_w:
            # No resize needed, just flatten
            pos_embed_flat = self.pos_embed.reshape(1, orig_h * orig_w, -1)
        else:
            # Resize if needed - directly using F.interpolate
            if self.pos_embed_ar_preserving:
                L = max(grid_size)
                _interp_size = L, L
            else:
                _interp_size = grid_size
            pos_embed_flat = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2).float(),  # B,C,H,W
                size=_interp_size,
                mode=self.pos_embed_interp_mode,
                align_corners=False,
                antialias=True,
            )[:, :, :grid_size[0], :grid_size[1]].flatten(2).transpose(1, 2)
        pos_embed_flat = pos_embed_flat.to(dtype=x.dtype)

        x.add_(pos_embed_flat)

    @disable_compiler
    def _apply_factorized_naflex_pos_embed(
            self,
            x: torch.Tensor,
            patch_coord: torch.Tensor,
    ) -> None:
        """Apply factorized position embeddings to NaFlex batch in-place.

        Uses separate Y and X position embedding tables that are interpolated
        and combined for each sample's grid size.

        Args:
            x: Input tensor to add position embeddings to [B, N, C]
            patch_coord: Patch coordinates [B, N, 2] with (y, x) values
        """
        # Calculate grid sizes from patch coordinates
        naflex_grid_sizes = calculate_naflex_grid_sizes(patch_coord)
        assert len(naflex_grid_sizes) == x.size(0)   # one (H,W) per sample

        # Handle each batch element separately with its own grid size
        orig_h, orig_w = self.pos_embed_y.shape[1], self.pos_embed_x.shape[1]

        # bucket samples that share the same (H, W) so we build each grid once
        size_to_indices: Dict[Tuple[int, int], List[int]] = {}
        for bi, k in enumerate(naflex_grid_sizes):
            size_to_indices.setdefault(k, []).append(bi)

        def _interp1d(table: torch.Tensor, new_length: int, orig_length: int) -> torch.Tensor:
            """
            Resample a 1-D positional-embedding table to specified length
            and return it in (1, L, C) layout, dtype matching x.
            """
            if new_length == orig_length:
                return table.to(dtype=x.dtype)
            return F.interpolate(
                table.permute(0, 2, 1).float(),  # (1,C,L) → (1,C,L_out)
                size=new_length,
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1).to(dtype=x.dtype)  # → (1,L_out,C)

        for k, batch_indices in size_to_indices.items():
            target_h, target_w = k
            if self.pos_embed_ar_preserving:
                len_y = len_x = max(target_h, target_w)
            else:
                len_y, len_x = target_h, target_w

            pe_y = _interp1d(self.pos_embed_y, len_y, orig_h)[:, :target_h]  # (1,H,C)
            pe_x = _interp1d(self.pos_embed_x, len_x, orig_w)[:, :target_w]  # (1,W,C)

            # Broadcast, add and flatten to sequence layout (row major)
            pos = pe_y.unsqueeze(2) + pe_x.unsqueeze(1)        # (1,H,W,C)
            pos = pos.flatten(1, 2)

            seq_len = min(x.shape[1], pos.shape[1])
            x[:, :seq_len].index_add_(
                0,
                torch.as_tensor(batch_indices, device=x.device),
                pos[:, :seq_len].expand(len(batch_indices), -1, -1)
            )

    @disable_compiler
    def _apply_factorized_naflex_pos_embed_grid_sample(
            self,
            x: torch.Tensor,
            patch_coord: torch.Tensor,
    ) -> None:
        """Apply factorized position embeddings to NaFlex batch using grid_sample.

        Uses F.grid_sample for efficient interpolation of separate Y and X position
        embedding tables based on patch coordinates. Based on proposal by https://github.com/stas-sl

        Args:
            x: Input tensor to add position embeddings to [B, N, C]
            patch_coord: Patch coordinates [B, N, 2] with (y, x) values
        """
        device = x.device
        B, _, C = x.shape
        shapes = patch_coord.amax(dim=1) + 1

        if self.pos_embed_ar_preserving:
            # Aspect ratio preserving mode: use square grid with uniform scaling
            L_i = shapes.amax(dim=1)  # (B,) max(h_i, w_i)
            L_global = L_i.amax()
            grid_size_y = grid_size_x = L_global
            scale_x = scale_y = L_global / L_i  # uniform zoom (B,)
        else:
            # Standard mode: different scaling for x and y
            grid_size_y, grid_size_x = shapes.amax(0)
            scale_x = grid_size_x / shapes[:, 1]  # horizontal zoom (B,)
            scale_y = grid_size_y / shapes[:, 0]  # vertical zoom (B,)

        def _interp1d(table: torch.Tensor, scale: torch.Tensor, out_length: torch.Tensor) -> torch.Tensor:
            pe = table.permute(0, 2, 1).unsqueeze(2).expand(B, -1, -1, -1).float()  # (1, L, C) -> (B, C, 1, L)
            theta = torch.zeros(B, 2, 3, device=x.device)
            theta[:, 0, 0] = scale
            theta[:, 0, 2] = scale - 1
            theta[:, 1, 1] = 1
            grid = F.affine_grid(theta, (B, C, 1, out_length), align_corners=False)
            pe = F.grid_sample(pe, grid, mode='bilinear', align_corners=False, padding_mode='border')
            return pe.to(x.dtype)

        # Interpolate along each axis
        pe_x = _interp1d(self.pos_embed_x, scale=scale_x, out_length=grid_size_x)
        pe_y = _interp1d(self.pos_embed_y, scale=scale_y, out_length=grid_size_y)

        bi = torch.arange(B, device=device).unsqueeze(1)
        x += pe_x[bi, :, 0, patch_coord[..., 1]] + pe_y[bi, :, 0, patch_coord[..., 0]]

    def _apply_factorized_pos_embed(
            self,
            x: torch.Tensor,
            grid_size: List[int],
    ) -> None:
        """Apply factorized position embeddings to standard 2D batch in-place.

        Uses separate Y and X position embedding tables that are interpolated
        and combined for the specified grid size.

        Args:
            x: Input tensor to add position embeddings to [B, H*W, C]
            grid_size: Target grid size as [height, width]
        """
        orig_h, orig_w = self.pos_embed_y.shape[1], self.pos_embed_x.shape[1]
        target_h, target_w = grid_size

        if self.pos_embed_ar_preserving:
            len_y = len_x = max(target_h, target_w)
        else:
            len_y, len_x = target_h, target_w

        def _interp1d(table: torch.Tensor, new_length: int, orig_length: int) -> torch.Tensor:
            if new_length == orig_length:
                return table.to(dtype=x.dtype)
            return F.interpolate(
                table.permute(0, 2, 1).float(),  # (1,L,C) -> (1,C,L)
                size=new_length,
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1).to(dtype=x.dtype)  # (1,L,C)

        # Interpolate embeddings
        pe_y = _interp1d(self.pos_embed_y, len_y, orig_h)[:, :target_h]  # (1,H,C)
        pe_x = _interp1d(self.pos_embed_x, len_x, orig_w)[:, :target_w]  # (1,W,C)

        # Broadcast, add and flatten to sequence layout (row major)
        pos_embed = pe_y.unsqueeze(2) + pe_x.unsqueeze(1)  # (1, H, W, C)
        pos_embed_flat = pos_embed.flatten(1, 2)  # (1, H*W, C)

        x.add_(pos_embed_flat)

    def forward(
            self,
            x: torch.Tensor,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        """Forward pass for patch embedding with position encoding.

        Args:
            x: Input tensor. Supported formats:
                - [B, C, H, W] for conv mode
                - [B, N, P*P*C] for pre-patchified linear mode (normal)
                - [B, N, Ph, Pw, C] for pre-patchified linear mode (variable patch size)
            patch_coord: Optional patch coordinates [B, N, 2] for NaFlex mode.
            patch_valid: Optional validity mask for patches [B, N] for NaFlex mode.

        Returns:
            Tuple of (embedded_tensor, grid_size) where:
                - embedded_tensor: [B, num_prefix_tokens + N, embed_dim]
                - grid_size: (H, W) tuple for standard mode, None for NaFlex mode
        """
        grid_size: Optional[Tuple[int, int]] = None
        B = x.shape[0]
        if self.is_linear:
            # Linear embedding path, works with NaFlex mode or standard 2D mode
            if patch_coord is None:
                # Standard 2D (B, C, H, W) mode
                _assert(x.ndim == 4, 'Expecting 2D image input with input ndim == 4')
                x, grid_size = batch_patchify(x, self.patch_size, pad=self.dynamic_img_pad)
            else:
                # Pre-patchified NaFlex mode
                # Variable patch size mode: [B, N, Ph, Pw, C], normal mode: [B, N, P*P*C]
                _assert(x.ndim == 5 or x.ndim == 3, 'Expecting patchified input with ndim == 3 or 5.')

            # Handle variable patch size projection
            if self.enable_patch_interpolator and x.ndim == 5:
                _assert(self.norm_input is None, 'input norm not supported with patch resizing')

                # Apply projection with interpolation
                x = self.patch_interpolator(
                    x,
                    self.proj.weight,
                    self.proj.bias,
                    patch_size=tuple(x.shape[2:4]),  # patch size from [B, N, Ph, Pw, C] shape
                    is_linear=True,
                )
            else:
                # Standard projection
                x = x.flatten(2)  # ensure [B, N, P*P*C], flatten Ph*Pw*C if separate
                if self.norm_input is not None:
                    x = self.norm_input(x)
                x = self.proj(x)
        else:
            _assert(x.ndim == 4, 'Convolutional input must be 4D')
            if self.dynamic_img_pad:
                H, W = x.shape[-2:]
                pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
                pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
                x = F.pad(x, (0, pad_w, 0, pad_h))

            x = self.proj(x)

            grid_size = x.shape[-2:]
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC

        # Apply normalization after flattening
        x = self.norm(x)

        if self.pos_embed_type == 'learned':
            if grid_size is not None:
                # Standard 2D mode
                self._apply_learned_pos_embed(x, grid_size=grid_size)
            else:
                # NaFlex mode
                if self.pos_embed_use_grid_sample:
                    self._apply_learned_naflex_pos_embed_grid_sample(x, patch_coord=patch_coord)
                else:
                    self._apply_learned_naflex_pos_embed(x, patch_coord=patch_coord)
        elif self.pos_embed_type == 'factorized':
            if grid_size is not None:
                # Standard 2D mode
                self._apply_factorized_pos_embed(x, grid_size=grid_size)
            else:
                # NaFlex mode
                if self.pos_embed_use_grid_sample:
                    self._apply_factorized_naflex_pos_embed_grid_sample(x, patch_coord=patch_coord)
                else:
                    self._apply_factorized_naflex_pos_embed(x, patch_coord=patch_coord)

        # Prepare and add class and register tokens
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(B, -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(B, -1, -1))
        # Add tokens to the beginning
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        # Apply dropout
        x = self.pos_drop(x)

        return x, grid_size


@register_notrace_function
def create_attention_mask(
        patch_valid: torch.Tensor,
        num_prefix_tokens: int = 0,
        symmetric: bool = True,
        q_len: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Creates an attention mask from patch validity information.

    Supports two modes controlled by `symmetric`:
    1. `symmetric=True` (default): Creates a symmetric mask of shape
       [B, 1, seq_len, seq_len]. An attention pair (i, j) is allowed only if
       both token i and token j are valid. Suitable for standard self-attention.
    2. `symmetric=False`: Creates a potentially non-square mask of shape
       [B, 1, q_len, kv_len]. An attention pair (q, k) is allowed only if
       the key/value token k is valid. Query token validity is not checked
       in the mask itself. Useful for cross-attention or specific self-attention
       implementations `q_len` can be specified.

    Used for NaFlex mode to handle variable token counts and padding tokens.

    Args:
        patch_valid: Tensor of shape [B, N] with True for valid patches, False for padding.
        num_prefix_tokens: Number of prefix tokens (class token, register tokens)
            to prepend, which are always considered valid.
        symmetric: If True, create a symmetric mask.
            If False, create an expanded mask based only on key/value validity.
        q_len: Query sequence length override. Only used when `symmetric` is False.
            Defaults to the key/value sequence length (`kv_len`) if None.
        dtype: Dtype of the output attention mask (e.g., torch.float32).

    Returns:
        Attention mask tensor. Additive mask (-inf for masked, 0 for unmasked).
        Shape is [B, 1, seq_len, seq_len] if symmetric=True,
        or [B, 1, q_len, kv_len] if symmetric=False.
    """
    if patch_valid is None:
        return None

    patch_valid = patch_valid.bool() # Ensure boolean type
    B, N = patch_valid.shape
    kv_len = N # Initial key/value length is the number of patches

    # Prepend prefix tokens if any
    if num_prefix_tokens > 0:
        # Create prefix validity tensor on the same device/dtype base as patch_valid
        prefix_valid = patch_valid.new_ones((B, num_prefix_tokens), dtype=torch.bool)
        # Concatenate prefix and patch validity. Shape becomes [B, num_prefix_tokens + N]
        patch_valid = torch.cat([prefix_valid, patch_valid], dim=1)
        kv_len += num_prefix_tokens # Update total key/value sequence length

    if symmetric:
        # Symmetric mask is True where BOTH query and key are valid
        mask_bool = patch_valid.unsqueeze(-1) & patch_valid.unsqueeze(1)
        mask_bool = mask_bool.unsqueeze(1)  # Add head dimension: [B, 1, seq_len, seq_len]
    else:
        # Expanded mask
        q_len = q_len or kv_len
        mask_bool = patch_valid[:, None, None, :].expand(B, 1, q_len, kv_len)

    # Create the float mask and apply masking using additive mask convention
    mask_float = torch.zeros_like(mask_bool, dtype=dtype)
    # Fill with negative infinity where mask_bool is False (masked positions)
    mask_float.masked_fill_(~mask_bool, torch.finfo(dtype).min)

    return mask_float


@register_notrace_function
def global_pool_naflex(
        x: torch.Tensor,
        patch_valid: Optional[torch.Tensor] = None,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
) -> torch.Tensor:
    """Global pooling with NaFlex support for masked tokens.

    Applies global pooling while respecting patch validity masks to exclude
    padding tokens from pooling operations.

    Args:
        x: Input tensor with shape [B, N, C]
        patch_valid: Optional validity mask for patches [B, N-num_prefix_tokens]
        pool_type: Type of pooling ('token', 'avg', 'avgmax', 'max')
        num_prefix_tokens: Number of prefix tokens (class/register)
        reduce_include_prefix: Whether to include prefix tokens in pooling reduction

    Returns:
        Pooled tensor with shape [B, C]
    """
    if patch_valid is None or pool_type not in ('avg', 'avgmax', 'max'):
        # Fall back to standard pooling
        x = global_pool_nlc(
            x,
            pool_type=pool_type,
            num_prefix_tokens=num_prefix_tokens,
            reduce_include_prefix=reduce_include_prefix,
        )
        return x

    # For NaFlex mode, we need to apply masked pooling to exclude padding tokens
    if num_prefix_tokens > 0:
        if reduce_include_prefix:
            # Include prefix tokens in pooling - they are always considered valid
            # patch_valid only covers patch tokens, so create combined validity mask
            prefix_valid = patch_valid.new_ones(x.shape[0], num_prefix_tokens)
            patch_valid = torch.cat([prefix_valid, patch_valid], dim=1)
        else:
            # Exclude prefix tokens from pooling (default behavior)
            x = x[:, num_prefix_tokens:]

    patch_valid_float = patch_valid.to(x.dtype)
    if pool_type == 'avg':
        # Compute masked average pooling, sum valid tokens and divide by count of valid tokens
        masked_sums = (x * patch_valid_float.unsqueeze(-1)).sum(dim=1)
        valid_counts = patch_valid_float.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = masked_sums / valid_counts
        return pooled
    elif pool_type == 'avgmax':
        # For avgmax, compute masked average and masked max
        masked_sums = (x * patch_valid_float.unsqueeze(-1)).sum(dim=1)
        valid_counts = patch_valid_float.sum(dim=1, keepdim=True).clamp(min=1)
        masked_avg = masked_sums / valid_counts

        # For max pooling we set masked positions to large negative value
        masked_x = x.clone()
        masked_x[~patch_valid] = torch.finfo(masked_x.dtype).min
        masked_max = masked_x.amax(dim=1)

        # Combine average and max
        return 0.5 * (masked_avg + masked_max)
    elif pool_type == 'max':
        # For max pooling we set masked positions to large negative value
        masked_x = x.clone()
        masked_x[~patch_valid] = torch.finfo(masked_x.dtype).min
        return masked_x.amax(dim=1)
    else:
        assert False


class NaFlexVit(nn.Module):
    """NaFlexVit: Vision Transformer with NaFlex support for flexible input handling.

    A flexible implementation of Vision Transformer that supports:
    - Standard image classification with various pooling strategies
    - NaFlex functionality for variable aspect ratios and resolutions
    - Linear patch embedding for pre-patchified inputs
    - Multiple position embedding strategies (learned, factorized, rope)
    - Comprehensive attention masking for efficient batch processing
    - Encapsulated embedding and position encoding in FlexEmbeds module
    - Compatible with standard ViT checkpoints through checkpoint filtering
    """

    def __init__(
            self,
            cfg: Optional[NaFlexVitCfg] = None,
            in_chans: int = 3,
            num_classes: int = 1000,
            img_size: Optional[Union[int, Tuple[int, int]]] = None,
            **kwargs,
    ) -> None:
        """Initialize NaFlexVit model.

        Args:
            cfg: Model configuration. If None, uses default NaFlexVitCfg.
            in_chans: Number of input image channels.
            num_classes: Number of classification classes.
            img_size: Input image size (for backwards compatibility with classic vit).
            **kwargs: Additional config parameters to override cfg values.
        """
        super().__init__()

        # Initialize config
        cfg = cfg or NaFlexVitCfg()
        if kwargs:
            cfg = _overlay_kwargs(cfg, **kwargs)

        # Validate configuration
        assert cfg.global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert cfg.class_token or cfg.global_pool != 'token'
        assert cfg.pos_embed in ('', 'none', 'learned', 'factorized')

        # Resolve layer implementations
        norm_layer = get_norm_layer(cfg.norm_layer) or LayerNorm
        embed_norm_layer = get_norm_layer(cfg.embed_norm_layer)
        act_layer = get_act_layer(cfg.act_layer) or nn.GELU
        block_fn = get_block_fn(cfg)
        mlp_layer = cfg.mlp_layer or Mlp   # TODO: Support configurable mlp_layer via string lookup

        # Store instance variables
        self.num_classes = num_classes
        self.global_pool = cfg.global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = cfg.embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if cfg.class_token else 0
        self.num_prefix_tokens += cfg.reg_tokens
        self.num_reg_tokens = cfg.reg_tokens
        self.has_class_token = cfg.class_token
        self.pool_include_prefix = cfg.pool_include_prefix
        self.grad_checkpointing = False

        # Initialize embedding module (includes patch, position embedding, and class/reg tokens)
        # FlexEmbeds is always used - handles both linear and conv embedding
        self.embeds = NaFlexEmbeds(
            patch_size=cfg.patch_size,
            in_chans=in_chans,
            embed_dim=cfg.embed_dim,
            proj_type=cfg.embed_proj_type,
            proj_bias=not cfg.pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            class_token=cfg.class_token,
            reg_tokens=cfg.reg_tokens,
            default_img_size=img_size,
            dynamic_img_pad=cfg.dynamic_img_pad,
            pos_embed=cfg.pos_embed,
            pos_embed_grid_size=cfg.pos_embed_grid_size,
            pos_embed_interp_mode=cfg.pos_embed_interp_mode,
            pos_embed_ar_preserving=cfg.pos_embed_ar_preserving,
            pos_embed_use_grid_sample=cfg.pos_embed_use_grid_sample,
            proj_norm_layer=embed_norm_layer,
            pos_drop_rate=cfg.pos_drop_rate,
            enable_patch_interpolator=getattr(cfg, 'enable_patch_interpolator', False),
        )
        self.norm_pre = norm_layer(cfg.embed_dim) if cfg.pre_norm else nn.Identity()

        # ROPE position embeddings at model level
        self.rope: Optional[nn.Module] = None
        self.rope_is_mixed = False
        if cfg.rope_type and cfg.rope_type != 'none':
            from timm.layers.pos_embed_sincos import RotaryEmbeddingCat, RotaryEmbeddingMixed
            if cfg.rope_type == 'mixed':
                self.rope = RotaryEmbeddingMixed(
                    cfg.embed_dim,
                    depth=cfg.depth,
                    num_heads=cfg.num_heads,
                    temperature=cfg.rope_temperature,
                    feat_shape=None,  # Dynamic shapes for NaFlex
                    grid_indexing=cfg.rope_grid_indexing,
                )
                self.rope_is_mixed = True
            elif cfg.rope_type == 'axial':
                self.rope = RotaryEmbeddingCat(
                    cfg.embed_dim // cfg.num_heads,
                    temperature=cfg.rope_temperature,
                    in_pixels=False,
                    feat_shape=None,  # Dynamic shapes for NaFlex
                    ref_feat_shape=cfg.rope_ref_feat_shape,
                    grid_offset=cfg.rope_grid_offset,
                    grid_indexing=cfg.rope_grid_indexing,
                )
                self.rope_is_mixed = False
            else:
                raise ValueError(f"Unknown rope_type: {cfg.rope_type}")

        # Patch dropout
        if cfg.patch_drop_rate > 0:
            self.patch_drop = PatchDropoutWithIndices(
                cfg.patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = None

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.depth)]  # stochastic depth decay rule
        # Create transformer blocks
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                proj_bias=cfg.proj_bias,
                init_values=cfg.init_values,
                proj_drop=cfg.proj_drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(cfg.depth)
        ])

        # Feature info for downstream tasks
        patch_reduction = self.embeds.feat_ratio(as_scalar=True)
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=cfg.embed_dim, reduction=patch_reduction)
            for i in range(cfg.depth)
        ]

        self.norm = norm_layer(cfg.embed_dim) if cfg.final_norm and not cfg.fc_norm else nn.Identity()

        # Classifier Head
        if cfg.global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=cfg.attn_pool_num_heads or cfg.num_heads,
                mlp_ratio=cfg.attn_pool_mlp_ratio or cfg.mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None

        # Handle fc_norm default value
        fc_norm = cfg.fc_norm
        if fc_norm is None:
            fc_norm = cfg.global_pool == 'avg'
        self.fc_norm = norm_layer(cfg.embed_dim) if cfg.final_norm and fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(cfg.drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if cfg.weight_init != 'skip':
            self.init_weights(cfg.weight_init)
        if cfg.fix_init:
            self.fix_init_weight()

    def fix_init_weight(self) -> None:
        """Apply initialization weight fix with layer-wise scaling."""
        def rescale(param: torch.Tensor, _layer_id: int) -> None:
            with torch.no_grad():
                param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if hasattr(layer, 'attn'):
                rescale(layer.attn.proj.weight, layer_id + 1)
            if hasattr(layer, 'mlp'):
                rescale(layer.mlp.fc2.weight, layer_id + 1)
            if hasattr(layer, 'attn_out_proj'):
                rescale(layer.attn_out_proj.weight, layer_id + 1)
            if hasattr(layer, 'mlp_out_proj'):
                rescale(layer.mlp_out_proj.weight, layer_id + 1)


    def init_weights(self, mode: str = '') -> None:
        """Initialize model weights according to specified scheme.

        Args:
            mode: Initialization mode ('jax', 'jax_nlhb', 'moco', or '')
        """
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        # Custom loading for the new model structure
        from .vision_transformer import _load_weights as _orig_load_weights

        def _load_weights_adapter(model, checkpoint_path, prefix=''):
            """Adapter function to handle the different model structure"""
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Map original keys to new structure
            for k in list(state_dict.keys()):
                if k.startswith('cls_token'):
                    state_dict['embeds.' + k] = state_dict.pop(k)
                elif k.startswith('reg_token'):
                    state_dict['embeds.' + k] = state_dict.pop(k)
                elif k.startswith('pos_embed'):
                    state_dict['embeds.' + k] = state_dict.pop(k)
                elif k.startswith('patch_embed'):
                    state_dict['embeds.' + k[12:]] = state_dict.pop(k)

            return _orig_load_weights(model, state_dict, prefix)

        _load_weights_adapter(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """Get set of parameter names that should not have weight decay applied.

        Returns:
            Set of parameter names to skip during weight decay
        """
        skip_list = {'embeds.pos_embed', 'embeds.cls_token', 'embeds.reg_token'}
        if self.rope and hasattr(self.rope, 'no_weight_decay'):
            skip_list.update(self.rope.no_weight_decay())
        return skip_list

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        """Get parameter group matcher for optimizer parameter grouping.

        Args:
            coarse: Whether to use coarse-grained grouping

        Returns:
            Dictionary mapping group names to regex patterns
        """
        return dict(
            stem=r'^embeds',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing for memory efficiency.

        Args:
            enable: Whether to enable gradient checkpointing
        """
        self.grad_checkpointing = enable
        if hasattr(self.embeds, 'patch_embed') and hasattr(self.embeds.patch_embed, 'set_grad_checkpointing'):
            self.embeds.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classification head module.

        Returns:
            Classification head module
        """
        return self.head

    @disable_compiler
    def _generate_rope_naflex(
            self,
            x: torch.Tensor,
            patch_coord: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor], Any]:
        """Generate ROPE position embeddings for NaFlex batch with variable grid sizes.

        Args:
            x: Input tensor [B, N, C]
            patch_coord: Patch coordinates [B, N, 2] with (y, x) values

        Returns:
            ROPE embeddings:
            - Axial mode: Tensor of shape [B, 1, N, dim*2]
            - Mixed mode: List of tensors, each of shape [B, num_heads, N, dim], one per depth layer
            - Mixed mode with iterator: Iterator yielding tensors per depth
        """
        # Calculate grid sizes for each sample
        naflex_grid_sizes = calculate_naflex_grid_sizes(patch_coord)

        # Build ROPE embeddings for each unique grid size
        size_to_indices = {}
        unique_sizes = []
        for bi, grid_size in enumerate(naflex_grid_sizes):
            if grid_size not in size_to_indices:
                size_to_indices[grid_size] = []
                unique_sizes.append(grid_size)
            size_to_indices[grid_size].append(bi)

        B, N, C = x.shape
        seq_len = N - self.num_prefix_tokens

        if self.rope_is_mixed:
            # Use an iterator for Mixed mode, returns [batch_size, depth, num_heads, seq_len, dim]
            return NaFlexRopeIterator(
                self.rope,
                size_to_indices,
                unique_sizes,
                B,
                seq_len,
                x.dtype,
                x.device
            )

        # Axial mode: [batch_size, seq_len, dim*2]
        rope_embeds = torch.zeros(B, seq_len, self.rope.dim * 2, dtype=x.dtype, device=x.device)

        if hasattr(self.rope, 'get_batch_embeds'):
            # Batch mode - generate unique embeds from one grid and then assign
            unique_embeds = self.rope.get_batch_embeds(unique_sizes)
            for grid_size, embed, batch_indices in zip(unique_sizes, unique_embeds, size_to_indices.values()):
                h, w = grid_size
                actual_len = h * w
                for bi in batch_indices:
                    rope_embeds[bi, :actual_len] = embed[:actual_len]

        else:
            # Generate each unique size separately and assign
            for grid_size, bi in size_to_indices.items():
                rope_embed = self.rope.get_embed(shape=grid_size)
                h, w = grid_size
                actual_len = h * w
                rope_embeds[bi, :actual_len] = rope_embed[:actual_len]

        rope_embeds = rope_embeds.unsqueeze(1)

        return rope_embeds

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classification head with new number of classes and pooling.

        Args:
            num_classes: Number of classes for new classification head
            global_pool: Optional new global pooling type
        """
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _forward_embeds(
            self,
            x,
            patch_coord,
            patch_valid,
            attn_mask,
    ) -> Dict[str, torch.Tensor]:
        """ Forward pass through patch / abs pos / rope pos embeds and patch dropout
        """
        naflex_mode = patch_coord is not None

        # patch embed, abs pos embed, returns global grid size as calculated from 'standard' NCHW batches
        x, grid_size = self.embeds(
            x,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
        )

        # Generate ROPE embeddings at model level
        rope_embeds = None
        if self.rope is not None:
            if patch_coord is not None:
                # NaFlex mode - variable grid sizes
                rope_embeds = self._generate_rope_naflex(x, patch_coord)
            elif grid_size is not None:
                # Standard mode - fixed grid size
                rope_embeds = self.rope.get_embed(shape=grid_size)
            else:
                assert False, 'Expected one of patch_coord or grid_size to be valid'

        # Apply patch dropout with coordinated updates
        keep_indices: Optional[torch.Tensor] = None
        if self.training and self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            # keep_indices excludes prefix tokens, can use directly on patch_valid & rope embeds
            if patch_valid is not None:
                patch_valid = patch_valid.gather(1, keep_indices)
            if rope_embeds is not None and not self.rope_is_mixed:
                # Update ROPE embeddings to match dropped tokens (only for axial mode)
                # Batch dim already present in NaFlex mode, but will be added in standard mode.
                rope_embeds = apply_keep_indices_nlc(x, rope_embeds, keep_indices, pos_embed_has_batch=naflex_mode)
                if not naflex_mode:
                    # B, N, dim -> B, 1, N, dim. Need head dim added for standard mode, already added in NaFlex.
                    rope_embeds = rope_embeds.unsqueeze(1)

        # Create attention mask from patch_valid after patch dropout applied
        if attn_mask is None:
            attn_mask = create_attention_mask(
                patch_valid,
                num_prefix_tokens=self.num_prefix_tokens,
                dtype=x.dtype
            )

        x = self.norm_pre(x)
        return {
            'patches': x,
            'patch_valid': patch_valid,
            'rope_embeds': rope_embeds,
            'attn_mask': attn_mask,
            'keep_indices': keep_indices,
        }

    def forward_intermediates(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            output_dict: bool = False,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
            output_dict: Return outputs as a dictionary with 'image_features' and 'image_intermediates' keys
            patch_coord: Optional patch coordinates [B, N, 2] for NaFlex mode
            patch_valid: Optional patch type indicators (1=patch, 0=padding) for NaFlex
            attn_mask: Optional attention mask for masked attention
        Returns:
            A tuple with (final_features, intermediates), a list of intermediate features, or a dictionary containing
            'image_features' and 'image_intermediates' (and optionally 'image_intermediates_prefix')
        """

        # FIXME unfinished / untested

        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        if isinstance(x, Dict):
            # Handle dictionary input from NaFlex collator
            patch_coord = x['patch_coord']
            patch_valid = x['patch_valid']
            patches = x['patches']
            assert False, 'WIP, patch mode needs more work'
        else:
            patches = x
            height, width = x.shape[-2:]
            H, W = self.embeds.dynamic_feat_size((height, width))

        # Forward pass through patch and abs position embedding
        embeds = self._forward_embeds(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
        )
        x = embeds['patches']
        rope_embeds = embeds.get('rope_embeds', None)
        keep_indices = embeds.get('keep_indices', None)
        attn_mask = embeds.get('attn_mask', None)

        # Forward pass through blocks
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]

        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        if self.rope_is_mixed and rope_embeds is not None:
            # Mixed mode with per-layer embeddings (list or iterator)
            for i, (blk, rope_embed) in enumerate(zip(self.blocks, rope_embeds)):
                # Apply patch dropout to rope_embed if needed
                if self.training and self.patch_drop is not None and keep_indices is not None:
                    # Apply patch dropout to rope_embed if needed (batch dim already present in naflex mode)
                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=embeds.get('naflex_mode', False),
                    )
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rope_embed, attn_mask=attn_mask)
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask)
                if i in take_indices:
                    # normalize intermediates with final norm layer if enabled
                    intermediates.append(self.norm(x) if norm else x)
        else:
            for i, blk in enumerate(blocks):
                # Axial ROPE mode with shared embeddings
                if rope_embeds is not None:
                    if do_checkpointing:
                        x = checkpoint(blk, x, rope=rope_embeds, attn_mask=attn_mask)
                    else:
                        x = blk(x, rope=rope_embeds, attn_mask=attn_mask)
                else:
                    if do_checkpointing:
                        x = checkpoint(blk, x, attn_mask=attn_mask)
                    else:
                        x = blk(x, attn_mask=attn_mask)
                if i in take_indices:
                    # normalize intermediates with final norm layer if enabled
                    intermediates.append(self.norm(x) if norm else x)

        # Process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            # reshape to BCHW output format
            intermediates = [
                y.reshape(y.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
                for y in intermediates
            ]

        # FIXME always use dict for NaFlex mode to return masks and more?

        # For dictionary output
        if output_dict:
            result_dict = {}
            # Intermediates are always included
            result_dict['image_intermediates'] = intermediates
            if prefix_tokens is not None and return_prefix_tokens:
                result_dict['image_intermediates_prefix'] = prefix_tokens

            # Only include features if not intermediates_only
            if not intermediates_only:
                x_final = self.norm(x)
                result_dict['image_features'] = x_final

            return result_dict

        # For non-dictionary output, maintain the original behavior
        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def forward_features(
            self,
            patches: torch.Tensor,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        """
        naflex_mode = patch_coord is not None

        # Pass through patch & abs position embedding module with patch coordinate/type support
        embeds = self._forward_embeds(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
        )
        x = embeds['patches']
        rope_embeds = embeds.get('rope_embeds', None)
        keep_indices = embeds.get('keep_indices', None)
        attn_mask = embeds.get('attn_mask', None)

        # Apply transformer blocks with masked attention and/or ROPE if provided
        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        if self.rope_is_mixed and rope_embeds is not None:
            # Mixed mode with per-layer embeddings (list or iterator)
            for i, (blk, rope_embed) in enumerate(zip(self.blocks, rope_embeds)):
                if self.training and self.patch_drop is not None and keep_indices is not None:
                    # Apply patch dropout to rope_embed if needed (batch dim already present in naflex mode)
                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=naflex_mode,
                    )
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rope_embed, attn_mask=attn_mask)
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask)
        elif rope_embeds is not None:
            # Axial ROPE mode with shared embeddings
            for blk in self.blocks:
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rope_embeds, attn_mask=attn_mask)
                else:
                    x = blk(x, rope=rope_embeds, attn_mask=attn_mask)
        else:
            for blk in self.blocks:
                if do_checkpointing:
                    x = checkpoint(blk, x, attn_mask=attn_mask)
                else:
                    x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)

        if naflex_mode:
            return {
                'patches': x,
                'patch_valid': embeds.get('patch_valid', None),
            }

        return x

    def _pool(
            self,
            x: torch.Tensor,
            pool_type: Optional[str] = None,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.attn_pool is not None:
            attn_mask = create_attention_mask(
                patch_valid,
                num_prefix_tokens=self.num_prefix_tokens if self.pool_include_prefix else 0,
                symmetric=False,
                q_len=1,
                dtype=x.dtype,
            )
            if not self.pool_include_prefix:
                x = x[:, self.num_prefix_tokens:]
            x = self.attn_pool(x, attn_mask=attn_mask)
            return x

        pool_type = self.global_pool if pool_type is None else pool_type

        x = global_pool_naflex(
            x,
            patch_valid,
            pool_type=pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )
        return x

    def forward_head(
            self,
            patches: torch.Tensor,
            pre_logits: bool = False,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._pool(patches, patch_valid=patch_valid)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional NaFlex support.

        Args:
            x: Input tensor. Supported formats:
                - [B, C, H, W] standard image input
                - [B, N, P*P*C] pre-patchified tensor (flattened patches)
                - [B, N, Ph, Pw, C] pre-patchified tensor (variable patch size)
                - Dict from NaFlex collator
            patch_coord: Optional patch coordinates [B, N, 2] for NaFlex mode.
            patch_valid: Optional patch validity indicators for NaFlex.
            attn_mask: Optional attn mask to override defaults generated from patch_valid

        Returns:
            Model output tensor.
        """
        input_is_dict = isinstance(x, Dict)
        naflex_mode = input_is_dict or patch_coord is not None
        if naflex_mode:
            if input_is_dict:
                # Handle dictionary input from NaFlex collator, dict inputs take priority over args
                patches = x['patches']
                patch_valid = x.get('patch_valid', patch_valid)
                patch_coord = x.get('patch_coord', patch_coord)
                attn_mask = x.get('attn_mask', attn_mask)
            else:
                patches = x
            _assert(patch_coord is not None, "patch_coord is required in naflex mode")
            _assert(patch_valid is not None, "patch_valid is required in naflex mode")

            features = self.forward_features(
                patches=patches,
                patch_valid=patch_valid,
                patch_coord=patch_coord,
                attn_mask=attn_mask,
            )

            # Pass patches & patch_valid to forward_head for masked pooling
            x = self.forward_head(**features)
        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
        return x


def _debug_dump_patches(x):
    # DEBUG, reconstruct patches & save
    patch_coord = x['patch_coord']
    patch_valid = x['patch_valid']
    patches = x['patches']
    for i in range(len(patches)):
        patch = patches[i][patch_valid[i]]
        h = (patch_coord[i, :, 0].max() + 1).item()
        w = (patch_coord[i, :, 1].max() + 1).item()
        patch = patch.reshape(h, w, 16, 16, 3).permute(4, 0, 2, 1, 3)
        patch = patch.reshape(3, h*16, w*16)
        from torchvision.utils import save_image
        save_image(patch, f'patch_{i}.jpg', normalize=True)


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> Callable:
    """Function imported from vision_transformer.py to maintain compatibility"""
    from .vision_transformer import init_weights_vit_jax, init_weights_vit_moco, init_weights_vit_timm

    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def checkpoint_filter_fn(state_dict: Dict[str, Any], model: NaFlexVit) -> Dict[str, Any]:
    """Handle state dict conversion from original ViT to the new version with combined embedding."""

    # Handle CombinedEmbed module pattern
    out_dict = {}
    for k, v in state_dict.items():
        # Convert tokens and embeddings to combined_embed structure
        if k == 'pos_embed':
            # Handle position embedding format conversion - from (1, N, C) to (1, H, W, C)
            if hasattr(model.embeds, 'pos_embed') and v.ndim == 3:
                num_cls_token = 0
                num_reg_token = 0
                if 'reg_token' in state_dict:
                    num_reg_token = state_dict['reg_token'].shape[1]
                if 'cls_token' in state_dict:
                    num_cls_token = state_dict['cls_token'].shape[1]
                num_prefix_tokens = num_cls_token + num_reg_token

                # Original format is (1, N, C), need to reshape to (1, H, W, C)
                num_patches = v.shape[1]
                num_patches_no_prefix = num_patches - num_prefix_tokens
                grid_size_no_prefix = math.sqrt(num_patches_no_prefix)
                grid_size = math.sqrt(num_patches)
                if (grid_size_no_prefix != grid_size
                        and (grid_size_no_prefix.is_integer() and not grid_size.is_integer())
                ):
                    # make a decision, did the pos_embed of the original include the prefix tokens?
                    num_patches = num_patches_no_prefix
                    cls_token_emb = v[:, 0:num_cls_token]
                    if cls_token_emb.numel():
                        state_dict['cls_token'] += cls_token_emb
                    reg_token_emb = v[:, num_cls_token:num_reg_token]
                    if reg_token_emb.numel():
                        state_dict['reg_token'] += reg_token_emb
                    v = v[:, num_prefix_tokens:]
                    grid_size = grid_size_no_prefix
                grid_size = int(grid_size)

                # Check if it's a perfect square for a standard grid
                if grid_size * grid_size == num_patches:
                    # Reshape from (1, N, C) to (1, H, W, C)
                    v = v.reshape(1, grid_size, grid_size, v.shape[2])
                else:
                    # Not a square grid, we need to get the actual dimensions
                    if hasattr(model.embeds.patch_embed, 'grid_size'):
                        h, w = model.embeds.patch_embed.grid_size
                        if h * w == num_patches:
                            # We have the right dimensions
                            v = v.reshape(1, h, w, v.shape[2])
                        else:
                            # Dimensions don't match, use interpolation
                            _logger.warning(
                                f"Position embedding size mismatch: checkpoint={num_patches}, model={(h * w)}. "
                                f"Using default initialization and will resize in forward pass."
                            )
                            # Keep v as is, the forward pass will handle resizing

            out_dict['embeds.pos_embed'] = v
        elif k == 'cls_token':
            out_dict['embeds.cls_token'] = v
        elif k == 'reg_token':
            out_dict['embeds.reg_token'] = v
        # Convert patch_embed.X to embeds.patch_embed.X
        elif k.startswith('patch_embed.'):
            suffix = k[12:]
            if suffix == 'proj.weight':
                v = v.permute(0, 2, 3, 1).flatten(1)
            new_key = 'embeds.' + suffix
            out_dict[new_key] = v
        else:
            out_dict[k] = v

    return out_dict


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 384, 384),
        'pool_size': None,
        'crop_pct': 1.0,
        'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'embeds.proj',
        'classifier': 'head',
        'license': 'apache-2.0',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'naflexvit_base_patch16_gap.e300_s576_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'naflexvit_base_patch16_par_gap.e300_s576_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'naflexvit_base_patch16_parfac_gap.e300_s576_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'naflexvit_base_patch16_map.untrained': _cfg(),

    'naflexvit_base_patch16_siglip.untrained': _cfg(),
    'naflexvit_so400m_patch16_siglip.untrained': _cfg(),
})


def _create_naflexvit(variant: str, pretrained: bool = False, **kwargs) -> NaFlexVit:
    out_indices = kwargs.pop('out_indices', 3)
    cfg = kwargs.pop('cfg', NaFlexVitCfg())
    cfg_field_names = {f.name for f in fields(NaFlexVitCfg)}
    # pop in-place so the original kwargs is emptied of cfg-specific keys
    cfg_updates = {k: kwargs.pop(k) for k in list(kwargs) if k in cfg_field_names}
    if cfg_updates:
        cfg = _overlay_kwargs(cfg, **cfg_updates)

    model = build_model_with_cfg(
        NaFlexVit, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        cfg=cfg,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def _create_naflexvit_from_classic(
        variant: str,
        pretrained: bool = False,
        **kwargs,
) -> NaFlexVit:
    """Create FlexVit model from classic VisionTransformer configuration.

    This function handles the parameter mapping and configuration logic needed
    to create FlexVit models that are compatible with classic VisionTransformer
    configurations and pretrained weights.

    Args:
        variant: Model variant name
        pretrained: Whether to load pretrained weights
        **kwargs: Classic VisionTransformer parameters

    Returns:
        FlexVit model instance
    """
    # Remove VisionTransformer-specific parameters that don't apply to FlexVit
    kwargs.pop('no_embed_class', None)
    kwargs.pop('dynamic_img_size', None)

    # Handle global pooling and fc_norm defaults that differ between ViT and FlexVit
    gp = kwargs.pop('global_pool', 'token')  # Original ViTs default to cls token pooling
    fc_norm = kwargs.pop('fc_norm', None)    # Original ViTs used fc_norm when not set and avg pooling used
    if fc_norm is None and gp == 'avg':
        fc_norm = True

    # Set FlexVit-specific defaults that differ from VisionTransformer
    flex_kwargs = {
        'pos_embed_grid_size': None,  # rely on img_size (// patch_size) that will be passed through
        'class_token': kwargs.get('class_token', True),
        'global_pool': gp,
        'fc_norm': fc_norm,
        'scale_mlp_norm': kwargs.pop('scale_mlp_norm', False),
        'scale_attn_inner_norm': kwargs.pop('scale_attn_norm', False),
        **kwargs  # User overrides take precedence
    }

    return _create_naflexvit(variant, pretrained, **flex_kwargs)


def _create_naflexvit_from_eva(
        variant: str,
        pretrained: bool = False,
        **kwargs,
) -> NaFlexVit:
    """Create NaFlexVit model from EVA configuration.

    This function handles the parameter mapping and configuration logic needed
    to create NaFlexVit models that are compatible with EVA configurations
    and pretrained weights.

    Args:
        variant: Model variant name
        pretrained: Whether to load pretrained weights
        **kwargs: EVA model parameters

    Returns:
        NaFlexVit model instance
    """
    # Handle EVA's unique parameters & block args
    kwargs.pop('no_embed_class', None)  # EVA specific, not used in NaFlexVit (always no-embed)

    # Map EVA's rope parameters
    use_rot_pos_emb = kwargs.pop('use_rot_pos_emb', False)
    rope_mixed_mode = kwargs.pop('rope_mixed_mode', False)
    rope_temperature = kwargs.pop('rope_temperature', 10000.)
    rope_grid_offset = kwargs.pop('rope_grid_offset', 0.)
    rope_grid_indexing = kwargs.pop('rope_grid_indexing', 'ij')
    if use_rot_pos_emb:
        rope_type = 'mixed' if rope_mixed_mode else 'axial'
    else:
        rope_type = 'none'

    # Handle norm/pool resolution logic to mirror EVA
    gp = kwargs.pop('global_pool', 'avg')
    use_pre_transformer_norm = kwargs.pop('use_pre_transformer_norm', False)
    use_post_transformer_norm = kwargs.pop('use_post_transformer_norm', True)
    use_fc_norm = kwargs.pop('use_fc_norm', None)
    if use_fc_norm is None:
        use_fc_norm = gp == 'avg'  # default on if avg pool used

    # Set NaFlexVit-specific parameters
    naflex_kwargs = {
        'pos_embed_grid_size': None,  # rely on img_size (// patch_size)
        'class_token': kwargs.get('class_token', True),
        'reg_tokens':  kwargs.pop('num_reg_tokens', kwargs.get('reg_tokens', 0)),
        'global_pool': gp,
        'pre_norm': use_pre_transformer_norm,
        'final_norm': use_post_transformer_norm,
        'fc_norm': use_fc_norm,
        'pos_embed': 'learned' if kwargs.pop('use_abs_pos_emb', True) else 'none',
        'rope_type': rope_type,
        'rope_temperature': rope_temperature,
        'rope_grid_offset': rope_grid_offset,
        'rope_grid_indexing': rope_grid_indexing,
        'rope_ref_feat_shape': kwargs.get('ref_feat_shape', None),
        'attn_type': kwargs.pop('attn_type', 'eva'),
        'swiglu_mlp': kwargs.pop('swiglu_mlp', False),
        'qkv_fused': kwargs.pop('qkv_fused', True),
        'scale_mlp_norm': kwargs.pop('scale_mlp', False),
        'scale_attn_inner_norm': kwargs.pop('scale_attn_inner', False),
        **kwargs  # Pass remaining kwargs through
    }

    return _create_naflexvit(variant, pretrained, **naflex_kwargs)


@register_model
def naflexvit_base_patch16_gap(pretrained: bool = False, **kwargs) -> NaFlexVit:
    """ViT-Base with NaFlex functionality and global average pooling.
    """
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        global_pool='avg',
        reg_tokens=4,
        fc_norm=True,
    )
    model = _create_naflexvit('naflexvit_base_patch16_gap', pretrained=pretrained, cfg=cfg, **kwargs)
    return model


@register_model
def naflexvit_base_patch16_par_gap(pretrained: bool = False, **kwargs) -> NaFlexVit:
    """ViT-Base with NaFlex functionality, aspect preserving pos embed, global average pooling.
    """
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        pos_embed_ar_preserving=True,
        global_pool='avg',
        reg_tokens=4,
        fc_norm=True,
    )
    model = _create_naflexvit('naflexvit_base_patch16_par_gap', pretrained=pretrained, cfg=cfg, **kwargs)
    return model


@register_model
def naflexvit_base_patch16_parfac_gap(pretrained: bool = False, **kwargs) -> NaFlexVit:
    """ViT-Base with NaFlex functionality, aspect preserving & factorized pos embed, global average pooling.
    """
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        pos_embed_ar_preserving=True,
        pos_embed='factorized',
        global_pool='avg',
        reg_tokens=4,
        fc_norm=True,
    )
    model = _create_naflexvit('naflexvit_base_patch16_parfac_gap', pretrained=pretrained, cfg=cfg, **kwargs)
    return model


@register_model
def naflexvit_base_patch16_map(pretrained: bool = False, **kwargs) -> NaFlexVit:
    """ViT-Base with NaFlex functionality and MAP attention pooling.
    """
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        global_pool='map',
        reg_tokens=1,
    )
    model = _create_naflexvit('naflexvit_base_patch16_map', pretrained=pretrained, cfg=cfg, **kwargs)
    return model


@register_model
def naflexvit_so150m2_patch16_reg1_gap(pretrained: bool = False, **kwargs) -> NaFlexVit:
    """ViT-SO150M2 with NaFlex functionality for variable aspect ratios and resolutions.

    This model supports:
    1. Variable aspect ratios and resolutions via patch coordinates
    2. Position embedding interpolation for arbitrary grid sizes
    3. Explicit patch coordinates and valid token masking
    """
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=832,
        depth=21,
        num_heads=13,
        mlp_ratio=34/13,
        init_values=1e-5,
        qkv_bias=False,
        reg_tokens=1,
        global_pool='avg',
        fc_norm=True,
    )
    model = _create_naflexvit('naflexvit_so150m2_patch16_reg1_gap', pretrained=pretrained, cfg=cfg, **kwargs)
    return model


@register_model
def naflexvit_so150m2_patch16_reg1_map(pretrained: bool = False, **kwargs) -> NaFlexVit:
    """ViT-SO150M2 with NaFlex functionality for variable aspect ratios and resolutions.

    This model supports:
    1. Variable aspect ratios and resolutions via patch coordinates
    2. Position embedding interpolation for arbitrary grid sizes
    3. Explicit patch coordinates and valid token masking
    """
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=832,
        depth=21,
        num_heads=13,
        mlp_ratio=34/13,
        init_values=1e-5,
        qkv_bias=False,
        reg_tokens=1,
        global_pool='map',
    )
    model = _create_naflexvit('naflexvit_so150m2_patch16_reg1_map', pretrained=pretrained, cfg=cfg, **kwargs)
    return model


@register_model
def naflexvit_base_patch16_siglip(pretrained: bool = False, **kwargs) -> NaFlexVit:
    """ViT-Base with NaFlex functionality and SigLIP-style configuration.
    """
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        act_layer='gelu_tanh',
        global_pool='map',
    )
    model = _create_naflexvit('naflexvit_base_patch16_siglip', pretrained=pretrained, cfg=cfg, **kwargs)
    return model


@register_model
def naflexvit_so400m_patch16_siglip(pretrained: bool = False, **kwargs) -> NaFlexVit:
    """ViT-SO400M with NaFlex functionality for variable aspect ratios and resolutions.
    """
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7362,
        act_layer='gelu_tanh',
        global_pool='map',
    )
    model = _create_naflexvit('naflexvit_so400m_patch16_siglip', pretrained=pretrained, cfg=cfg, **kwargs)
    return model
