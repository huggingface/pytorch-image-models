""" Vision Transformer (New)

An improved version of the Vision Transformer with:
1. Encapsulated embedding and position encoding in a single module
2. Support for linear patch embedding on pre-patchified inputs
3. Support for NaFlex functionality (NaViT + FlexiViT)

Based on:
- Original Vision Transformer: https://arxiv.org/abs/2010.11929
- FlexiViT: https://arxiv.org/abs/2212.08013
- NaViT: https://arxiv.org/abs/2307.06304

Copyright 2025
"""

import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union, Final, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import AttentionPoolLatent, Mlp, to_2tuple, get_act_layer, get_norm_layer, LayerType, _assert
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._registry import register_model, generate_default_cfgs
from timm.models._manipulate import checkpoint_seq, named_apply

from .vision_transformer import Block, global_pool_nlc

_logger = logging.getLogger(__name__)


def batch_patchify(
        x: torch.Tensor,
        patch_size: Tuple[int, int],
        pad: bool = True,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, C, H, W = x.shape
    ph, pw = to_2tuple(patch_size)

    # Ensure the image is divisible by patch size
    if pad and (H % ph != 0 or W % pw != 0):
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        x = F.pad(x, (0, pad_w, 0, pad_h))

    nh, nw = H // ph, W // pw
    patches = x.view(B, C, nh, ph, nw, pw).permute(0, 2, 4, 3, 5, 1).reshape(B, nh * nw, ph * pw * C)

    return patches, (nh, nw)


class FlexEmbeds(nn.Module):
    """ Na(Flex) Embedding module for Vision Transformers

    This module encapsulates the complete embedding process for Vision Transformers:
    1. Patch embedding (via Conv2d or Linear)
    2. Class and register token preparation
    3. Position embedding addition
    4. Pre-normalization (if requested)
    5. Dropout application

    Also supports NaFlex functionality (NaViT + FlexiViT):
    - Variable aspect ratio and resolution via patch coordinates
    - Patch type indicators for handling padding tokens in attention
    - Flexible position embedding interpolation for arbitrary grid sizes

    Note: Only supports non-overlapping position and register tokens
    (i.e., position embeddings do not include class/register tokens)

    The patch embedding can be one of two types:
    1. Conv2d-based (default): For standard image inputs [B, C, H, W]
    2. Linear-based: For pre-patchified inputs [B, N, P*P*C]

    """

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            embed_layer: Optional[str] = None,  # 'conv' or 'linear', default is 'linear'
            input_norm_layer: Optional[Type[nn.Module]] = None,
            proj_norm_layer: Optional[Type[nn.Module]] = None,
            final_norm_layer: Optional[Type[nn.Module]] = None,
            pos_embed: str = 'learned',
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            class_token: bool = True,
            reg_tokens: int = 0,
            bias: bool = True,
            dynamic_img_pad: bool = False,
            pos_embed_grid_size: Optional[Tuple[int, int]] = (14, 14),
            pos_embed_interp_mode: str = 'bicubic',
            default_img_size: Union[int, Tuple[int, int]] = None,
    ):
        super().__init__()
        self.has_class_token = class_token
        self.num_reg_tokens = reg_tokens
        self.pos_embed_interp_mode = pos_embed_interp_mode
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.dynamic_img_pad = dynamic_img_pad

        # Calculate number of prefix tokens
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens

        # Create class and register tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None

        # Calculate grid size and number of patches
        self.default_img_size: Optional[Tuple[int, int]] = None
        self.pos_embed_grid_size: Optional[
            Tuple[int, int]] = None  # Stores the grid size used for learned pos embed init
        if pos_embed_grid_size is None and default_img_size is not None:
            self.default_img_size = to_2tuple(default_img_size)
            self.pos_embed_grid_size = tuple([s // p for s, p in zip(self.default_img_size, self.patch_size)])
        elif pos_embed_grid_size is not None:
            # Use provided pos_embed_grid_size for NaFlex mode
            self.pos_embed_grid_size = pos_embed_grid_size

        # Determine patch embedding type (linear or conv2d)
        if embed_layer == 'linear':
            # Create linear projection for pre-patchified inputs
            # Input dimension is patch_size^2 * in_chans
            patch_dim = self.patch_size[0] * self.patch_size[1] * in_chans
            self.norm_input = proj_norm_layer(patch_dim) if input_norm_layer else None
            self.proj = nn.Linear(patch_dim, embed_dim, bias=bias)
            self.flatten = False
            self.is_linear = True
        else:
            # Default to convolutional patch embedding for image inputs
            assert input_norm_layer is None
            self.norm_input = None
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
            )
            self.flatten = True
            self.is_linear = False

        # Create normalization layer after the projection
        self.norm_proj = proj_norm_layer(embed_dim) if proj_norm_layer else nn.Identity()

        # Create position embedding if needed - only for patches, never for prefix tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
            self.pos_embed_type = 'none'
        elif pos_embed == 'rope':
            self.pos_embed = None
            self.pos_embed_type = 'rope'
            # Rotary embeddings will be computed on-the-fly in the forward pass
        else:
            # Store position embedding in (1, H, W, dim) format for easier resizing
            if self.pos_embed_grid_size is not None:
                h, w = self.pos_embed_grid_size
                self.pos_embed = nn.Parameter(torch.randn(1, h, w, embed_dim) * .02)
                self.pos_embed_type = 'learned'
            else:
                raise ValueError("Cannot initialize position embeddings without grid_size. "
                                 "Please provide img_size or pos_embed_grid_size")

        # Pre-normalization layer (separate from the main norm layer)
        self.norm_final = final_norm_layer(embed_dim) if final_norm_layer is not None else nn.Identity()

        # Dropout layers
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            from timm.layers.patch_dropout import PatchDropout
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

    def feature_info(self, location):
        """Feature info utility method for feature extraction."""
        return dict(num_chs=self.embed_dim, reduction=self.patch_size)

    def feat_ratio(self, as_scalar=True):
        """Return the feature reduction ratio (stride)."""
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x, patch_coord=None, patch_valid=None):
        """Forward pass for combined embedding

        Args:
            x: Input tensor [B, C, H, W] or pre-patchified [B, N, P*P*C]
            patch_coord: Optional patch coordinates [B, N, 2] for NaFlex
            patch_valid: Optional patch type indicators (1=patch, 0=padding) for NaFlex

        Returns:
            Embedded tensor with position encoding and class/register tokens applied
            If patch_type is provided, also returns attention mask
        """
        # Apply patch embedding
        naflex_grid_sizes: Optional[List[Tuple[int, int]]] = None
        grid_size: Optional[Tuple[int, int]] = None

        if self.is_linear:
            # Linear embedding path, works with NaFlex mode or standard 2D mode
            B = x.shape[0]
            if x.ndim == 3:
                # pre-patchified NaFlex mode, input is expected to be (B, N, P*P*C) where N is num_patches
                _assert(patch_coord is not None, 'patch_coord must not be None in NaFlex mode')

                # Calculate the appropriate grid size from coords
                max_y = patch_coord[:, :, 0].max(dim=1)[0] + 1
                max_x = patch_coord[:, :, 1].max(dim=1)[0] + 1
                naflex_grid_sizes = [(h.item(), w.item()) for h, w in zip(max_y, max_x)]
            else:
                x, grid_size = batch_patchify(x, self.patch_size, pad=self.dynamic_img_pad)

            if self.norm_input is not None:
                x = self.norm_input(x)

            x = self.proj(x)
        else:
            assert x.ndim == 4, 'Convolutional input must be 4D'
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
        x = self.norm_proj(x)

        if self.pos_embed_type == 'learned':
            if naflex_grid_sizes:
                self._apply_learned_naflex_pos_embed(x, naflex_grid_sizes=naflex_grid_sizes)
            else:
                self._apply_learned_pos_embed(x, grid_size=grid_size)
        elif self.pos_embed_type == 'rope':
            assert False, "ROPE not yet implemented"

        # Prepare and add class and register tokens
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(B, -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(B, -1, -1))
        # Add tokens to the beginning
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        # Apply final pre-transformer normalization if specified
        x = self.norm_final(x)

        # Apply dropouts
        x = self.patch_drop(self.pos_drop(x))

        return x

    def _apply_learned_naflex_pos_embed(
            self,
            x: torch.Tensor,
            naflex_grid_sizes: List[Tuple[int, int]],
    ):
        orig_h, orig_w = self.pos_embed.shape[1:3]

        # Determine unique grid sizes
        size_to_indices = {}
        for bi, (h, w) in enumerate(naflex_grid_sizes):
            if not (h, w) in size_to_indices:
                size_to_indices[(h, w)] = [bi]
            else:
                size_to_indices[(h, w)].append(bi)

        # Handle each batch element separately with its own grid size
        for (h, w), batch_indices in size_to_indices.items():
            # Interpolate only once for this (h, w)
            if (h == orig_h) and (w == orig_w):
                pos_embed_flat = self.pos_embed.reshape(orig_h * orig_w, -1)
            else:
                pos_embed_resized = F.interpolate(
                    self.pos_embed.permute(0, 3, 1, 2),  # B,C,H,W
                    size=(h, w),
                    mode=self.pos_embed_interp_mode,
                    align_corners=False,
                    antialias=True,
                )
                pos_embed_flat = pos_embed_resized.permute(0, 2, 3, 1).reshape(h * w, -1)

            seq_len = min(x.shape[1], pos_embed_flat.shape[0])
            x[batch_indices, :seq_len].add_(pos_embed_flat[:seq_len])

    def _apply_learned_pos_embed(
            self,
            x: torch.Tensor,
            grid_size: Tuple[int, int],
    ):
        orig_h, orig_w = self.pos_embed.shape[1:3]
        if grid_size[0] != orig_h or grid_size[1] != orig_w:
            # Resize if needed - directly using F.interpolate
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),  # B,C,H,W
                size=grid_size,
                mode=self.pos_embed_interp_mode,
                align_corners=False,
                antialias=True,
            )
            # Convert back and flatten
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            pos_embed = pos_embed.reshape(1, grid_size[0] * grid_size[1], -1)

        else:
            # No resize needed, just flatten
            pos_embed = self.pos_embed.reshape(1, orig_h * orig_w, -1)

        x.add_(pos_embed)


def create_attention_mask(
        patch_valid: Optional[torch.Tensor],
        num_prefix_tokens: int = 0,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create attention mask from patch type information.

    Used for NaFlex mode to handle variable token counts and padding tokens.

    Args:
        patch_valid: Tensor of shape [B, N] with True for valid patches, False for padding
        num_prefix_tokens: Number of prefix tokens (class token, register tokens)
        dtype: Dtype of the attention mask

    Returns:
        Attention mask of shape [B, seq_len, seq_len] where seq_len = N + num_prefix_tokens,
        or None if patch_type is None
    """
    patch_valid = patch_valid.bool()
    B = patch_valid.shape[0]

    if num_prefix_tokens > 0:
        prefix_valid = patch_valid.new_ones((B, num_prefix_tokens))
        patch_valid = torch.cat([prefix_valid, patch_valid], dim=1)

    mask_bool = (patch_valid.unsqueeze(-1) & patch_valid.unsqueeze(1)).unsqueeze(1)
    mask_float = torch.zeros_like(mask_bool, dtype=dtype)
    mask_float.masked_fill_(~mask_bool, torch.finfo(mask_float.dtype).min)

    return mask_float

def create_attention_mask2(
    patch_valid: Optional[torch.Tensor],
    num_prefix_tokens: int = 0,
    q_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Create expanded attention mask from patch validity info.

    Used for NaFlex mode to handle variable token counts and padding tokens.

    Args:
        patch_valid: Tensor of shape [B, N] with True for valid patches, False for padding
        num_prefix_tokens: Number of prefix tokens (class token, register tokens)
        q_len: Length override for query sequence
        dtype: Dtype of the attention mask

    Returns:
        Attention mask of shape [B, seq_len, seq_len] where seq_len = N + num_prefix_tokens,
        or None if patch_type is None
    """
    patch_valid = patch_valid.bool()
    B, kv_len = patch_valid.shape

    if num_prefix_tokens > 0:
        prefix_valid = patch_valid.new_ones((B, num_prefix_tokens))
        patch_valid = torch.cat([prefix_valid, patch_valid], dim=1)
        kv_len = patch_valid.shape[1]

    q_len = q_len if q_len is not None else kv_len

    mask_bool = patch_valid[:, None, None, :].expand(B, 1, q_len, kv_len).to(dtype)
    mask_float = torch.zeros_like(mask_bool, dtype=dtype)
    mask_float.masked_fill_(~mask_bool, torch.finfo(mask_float.dtype).min)

    return mask_float


def create_pool_mask(
        patch_valid: Optional[torch.Tensor],
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    patch_valid = patch_valid.bool()
    mask_bool = patch_valid[:, None, None, :]
    mask_float = torch.zeros_like(mask_bool, dtype=dtype)
    mask_float.masked_fill_(~mask_bool, torch.finfo(mask_float.dtype).min)

    return mask_float


class VisionTransformerFlex(nn.Module):
    """ Vision Transformer (Na)Flex

    A flexible implementation of Vision Transformer with:
    1. Encapsulated embedding and position encoding in a single module
    2. Support for linear patch embedding on pre-patchified inputs
    3. Support for variable sequence length / aspect ratio images (NaFlex)
    """

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            init_values: Optional[float] = None,
            class_token: bool = False,
            reg_tokens: int = 0,
            pos_embed: str = 'learn',
            pos_embed_grid_size: Optional[Tuple[int, int]] = (16, 16),
            pos_embed_interp_mode: str = 'bicubic',
            default_img_size: Union[int, Tuple[int, int]] = 256,
            dynamic_img_pad: bool = False,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            num_classes: int = 1000,
            global_pool: str = 'map',
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            fix_init: bool = True,
            embed_layer_type: str = 'linear',
            embed_norm_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Args:
            patch_size: Patch size.
            in_chans: Number of image input channels.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            reg_tokens: Number of register tokens.
            pos_embed: Type of position embedding.
            pos_embed_grid_size: Size of position embedding grid.
            pos_embed_interp_mode: Interpolation mode for position embedding.
            default_img_size: Input image size.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer_type: Patch embedding implementation (e.g., 'linear', 'conv').
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.grad_checkpointing = False

        # Initialize embedding module (includes patch, position embedding, and class/reg tokens)
        # VisionTransformerEmbeds is always used - handles both linear and conv embedding
        self.embeds = FlexEmbeds(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            embed_layer=embed_layer_type,
            proj_norm_layer=embed_norm_layer,
            final_norm_layer=norm_layer if pre_norm else None,
            pos_embed=pos_embed,
            pos_embed_grid_size=pos_embed_grid_size,
            pos_embed_interp_mode=pos_embed_interp_mode,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            class_token=class_token,
            reg_tokens=reg_tokens,
            default_img_size=default_img_size,
            dynamic_img_pad=dynamic_img_pad,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])

        # Feature info for downstream tasks
        patch_reduction = to_2tuple(patch_size)
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=patch_reduction) for i in range(depth)]

        self.norm = norm_layer(embed_dim) if final_norm and not fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None

        self.fc_norm = norm_layer(embed_dim) if final_norm and fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
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
        skip_list = {'embeds.pos_embed', 'embeds.cls_token', 'embeds.reg_token'}
        return skip_list

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^embeds',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        if hasattr(self.embeds, 'patch_embed') and hasattr(self.embeds.patch_embed, 'set_grad_checkpointing'):
            self.embeds.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            output_dict: bool = False,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
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
            mask: Optional attention mask
        Returns:
            A tuple with (final_features, intermediates), a list of intermediate features, or a dictionary containing
            'image_features' and 'image_intermediates' (and optionally 'image_intermediates_prefix')
        """

        # FIXME unfinished / untested

        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # Create attention mask if patch_type is provided and mask is not
        if mask is None and patch_valid is not None:
            mask = create_attention_mask(patch_valid, self.num_prefix_tokens, x.dtype)

        # Forward pass through embedding
        x = self.embeds(x, patch_coord=patch_coord, patch_valid=patch_valid)

        # Forward pass through blocks
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
            
        for i, blk in enumerate(blocks):
            x = blk(x, attn_mask=mask)
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
            grid_size = self.embeds.pos_embed_grid_size
            if hasattr(self.embeds, 'dynamic_feat_size') and len(x.shape) >= 4:
                _, height, width, _ = x.shape if len(x.shape) == 4 else (None, *x.shape[-3:-1], None)
                H, W = self.embeds.dynamic_feat_size((height, width))
            else:
                H, W = grid_size
            intermediates = [y.reshape(y.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous() 
                           for y in intermediates]

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
            x: torch.Tensor,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pass through embedding module with patch coordinate/type support
        x = self.embeds(x, patch_coord=patch_coord, patch_valid=patch_valid)
        
        # Apply transformer blocks with masked attention if mask provided
        if attn_mask is not None:
            # We need to apply blocks one by one with mask
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
            
        x = self.norm(x)
        return x

    def _pool(
            self,
            x: torch.Tensor,
            pool_type: Optional[str] = None,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.attn_pool is not None:
            # For attention pooling, we need to pass the mask for NaFlex models
            attn_mask = create_pool_mask(patch_valid, dtype=x.dtype)
            x = self.attn_pool(x[:, self.num_prefix_tokens:], attn_mask=attn_mask)
            return x
        
        pool_type = self.global_pool if pool_type is None else pool_type
        
        # Handle padding mask for average pooling
        if patch_valid is not None and pool_type in ('avg', 'avgmax'):
            # For NaFlex mode, we need to apply masked pooling to exclude padding tokens
            # Extract only the patch part of the mask (excluding prefix tokens)
            if self.num_prefix_tokens > 0:
                # Apply the mask to extract only valid tokens
                x = x[:, self.num_prefix_tokens:]  # prefix tokens not included in pooling

            if pool_type == 'avg':
                # Compute masked average pooling
                # Sum valid tokens and divide by count of valid tokens
                masked_sums = (x * patch_valid.unsqueeze(-1).float()).sum(dim=1)
                valid_counts = patch_valid.float().sum(dim=1, keepdim=True).clamp(min=1)
                pooled = masked_sums / valid_counts
                return pooled
            elif pool_type == 'avgmax':
                # For avgmax, compute masked average and masked max
                # For max, we set masked positions to large negative value
                masked_sums = (x * patch_valid.unsqueeze(-1).float()).sum(dim=1)
                valid_counts = patch_valid.float().sum(dim=1, keepdim=True).clamp(min=1)
                masked_avg = masked_sums / valid_counts

                # For max pooling with mask
                masked_x = x.clone()
                masked_x[~patch_valid] = torch.finfo(masked_x.dtype).min
                masked_max = masked_x.max(dim=1)[0]

                # Combine average and max
                return 0.5 * (masked_avg + masked_max)

        # Fall back to standard pooling
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(
            self,
            x: torch.Tensor,
            pre_logits: bool = False,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._pool(x, patch_valid=patch_valid)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional NaFlex support.
        
        Args:
            x: Input tensor [B, C, H, W] or pre-patchified tensor [B, N, P*P*C] or NaFlex dict
            patch_coord: Optional patch coordinates [B, N, 2] for NaFlex mode
            patch_valid: Optional patch type indicators (1=patch, 0=padding) for NaFlex
            
        Returns:
            Model output tensor
        """
        # Handle dictionary input from NaFlex collator
        if isinstance(x, dict):
            assert patch_coord is None
            assert patch_valid is None
            # Extract the required components from the dictionary
            patch_coord = x['patch_coord']
            patch_valid = x['patch_valid']
            patches = x['patches']

            if False:
                # DEBUG, reconstruct patches
                for i in range(len(patches)):
                    patch = patches[i][patch_valid[i]]
                    h = (patch_coord[i, :, 0].max() + 1).item()
                    w = (patch_coord[i, :, 1].max() + 1).item()
                    patch = patch.reshape(h, w, 16, 16, 3).permute(4, 0, 2, 1, 3)
                    patch = patch.reshape(3, h*16, w*16)
                    from torchvision.utils import save_image
                    save_image(patch, f'patch_{i}.jpg', normalize=True)
        else:
            patches = x

        # Create attention mask if patch_type is provided
        if patch_valid is not None:
            attn_mask = create_attention_mask(
                patch_valid,
                num_prefix_tokens=self.num_prefix_tokens,
                dtype=patches.dtype
            )
        else:
            attn_mask = None

        # Forward features with mask
        x = self.forward_features(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
        )

        # Pass mask to forward_head for masked pooling
        x = self.forward_head(
            x,
            patch_valid=patch_valid,
        )
        return x


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> Callable:
    """Function imported from vision_transformer.py to maintain compatibility"""
    from .vision_transformer import init_weights_vit_jax, init_weights_vit_moco, init_weights_vit_timm
    
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 256, 256),
        'pool_size': None,
        'crop_pct': 0.95,
        'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'embeds.proj',
        'classifier': 'head',
        'license': 'apache-2.0',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'vit_naflex_base_patch16': _cfg(),
    'vit_naflex_base_patch16_gap': _cfg(),
    'vit_naflex_base_patch16_map': _cfg(),
})


def _create_vision_transformer_flex(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        VisionTransformerFlex, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )
    return model


@register_model
def vit_naflex_base_patch16(pretrained=False, **kwargs):
    """ViT-New with NaFlex functionality for variable aspect ratios and resolutions.
    
    This model supports:
    1. Variable aspect ratios and resolutions via patch coordinates
    2. Position embedding interpolation for arbitrary grid sizes
    3. Explicit patch coordinates and valid token masking
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer_flex(
        'vit_naflex_base_patch16', pretrained=pretrained, **model_args)
    return model


@register_model
def vit_naflex_base_patch16_gap(pretrained=False, **kwargs):
    """ViT-New with NaFlex functionality for variable aspect ratios and resolutions.
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        global_pool='avg', class_token=False, reg_tokens=4, **kwargs)
    model = _create_vision_transformer_flex(
        'vit_naflex_base_patch16_gap', pretrained=pretrained, **model_args)
    return model


@register_model
def vit_naflex_base_patch16_map(pretrained=False, **kwargs):
    """ViT-New with NaFlex functionality for variable aspect ratios and resolutions.
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, global_pool='map', **kwargs)
    model = _create_vision_transformer_flex(
        'vit_naflex_base_patch16_map', pretrained=pretrained, **model_args)
    return model


def checkpoint_filter_fn(state_dict, model):
    """Handle state dict conversion from original ViT to the new version with combined embedding."""
    from .vision_transformer import checkpoint_filter_fn as orig_filter_fn

    # FIXME conversion of existing vit checkpoints has not been finished or tested

    # Handle CombinedEmbed module pattern
    out_dict = {}
    for k, v in state_dict.items():
        # Convert tokens and embeddings to combined_embed structure
        if k == 'pos_embed':
            # Handle position embedding format conversion - from (1, N, C) to (1, H, W, C)
            if hasattr(model.embeds, 'pos_embed') and v.ndim == 3:
                # Original format is (1, N, C) - need to reshape to (1, H, W, C)
                num_patches = v.shape[1]
                grid_size = int(math.sqrt(num_patches))
                
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
            new_key = 'embeds.' + k[12:]
            out_dict[new_key] = v
        else:
            out_dict[k] = v
            
    # Call the original filter function to handle other patterns
    return orig_filter_fn(out_dict, model)