""" EVA

EVA ViT from https://github.com/baaivision/EVA , paper: https://arxiv.org/abs/2211.07636

@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang,
  Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}

EVA-02: A Visual Representation for Neon Genesis - https://arxiv.org/abs/2303.11331
@article{EVA02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}

@article{bolya2025perception,
  title={Perception encoder: The best visual embeddings are not at the output of the network},
  author={Bolya, Daniel and Huang, Po-Yao and Sun, Peize and Cho, Jang Hyun and Madotto, Andrea and Wei, Chen and Ma,
    Tengyu and Zhi, Jiale and Rajasegaran, Jathushan and Rasheed, Hanoona and others},
  journal={arXiv preprint arXiv:2504.13181},
  year={2025}
}

@inproceedings{heo2024rotary,
  title={Rotary position embedding for vision transformer},
  author={Heo, Byeongho and Park, Song and Han, Dongyoon and Yun, Sangdoo},
  booktitle={European Conference on Computer Vision},
  pages={289--305},
  year={2024},
  organization={Springer}
}

This file contains a number of ViT variants the utilise ROPE position embeddings, SwiGLU and other additions:
 * EVA & EVA02 model implementations that evolved from BEiT, additional models in vision_transformer.py.
 * `timm` original SBB ViT w/ ROPE position embeddings
 * Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)
 * ROPE-ViT from Naver AI (https://arxiv.org/abs/2403.13298)

Modifications by / Copyright 2023 Ross Wightman, original copyrights below
"""
# EVA models Copyright (c) 2022 BAAI-Vision
# EVA02 models Copyright (c) 2023 BAAI-Vision
import math
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import (
    PatchEmbed,
    Mlp,
    GluMlp,
    SwiGLU,
    LayerNorm,
    DropPath,
    PatchDropoutWithIndices,
    RotaryEmbeddingCat,
    RotaryEmbeddingMixed,
    apply_rot_embed_cat,
    apply_keep_indices_nlc,
    trunc_normal_,
    resample_patch_embed,
    resample_abs_pos_embed,
    global_pool_nlc,
    to_2tuple,
    use_fused_attn,
    AttentionRope,
    AttentionPoolLatent,
)
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model

__all__ = ['Eva']


class EvaAttention(nn.Module):
    """ EVA Attention with ROPE, no k-bias, and fused/unfused qkv options
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            qkv_bias_separate: bool = False,
            num_prefix_tokens: int = 1,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
            qk_norm: bool = False,
            scale_norm: bool = True,
    ):
        """
        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to add a bias term to the query, key, and value projections
            qkv_fused: Whether qkv projections are fused into one projection or separate
            qkv_bias_separate: Whether to apply bias to qkv as a separate addition or part of F.linear() call
            num_prefix_tokens: Number of reg/cls tokens at the beginning of the sequence that
                should not have position embeddings applied
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for the output projection
            attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
            norm_layer: Normalization layer constructor to use for QK and scale normalization
            qk_norm: Enable normalization of query (Q) and key (K) vectors with norm_layer
            scale_norm: Enable normalization (scaling) of attention output with norm_layer
        """
        super().__init__()
        if scale_norm or qk_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        attn_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()
        self.qkv_bias_separate = qkv_bias_separate

        if qkv_fused:
            self.qkv = nn.Linear(dim, attn_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(attn_dim))
                self.register_buffer('k_bias', torch.zeros(attn_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(attn_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, attn_dim, bias=False)
            self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(attn_dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the attention module.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            rope: Rotary position embeddings tensor for position-aware attention
            attn_mask: Optional attention mask to apply during attention computation

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1)

            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            attn_type: str = 'eva',
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
            **kwargs,
    ):
        """ Initialize the EVA transformer block.

        Args:
          dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias terms in query, key, value projections
            qkv_fused: Whether to use a single projection for query, key, value
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            swiglu_mlp: Whether to use SwiGLU activation in the MLP
            scale_mlp: Whether to use normalization in the MLP
            scale_attn_inner: Whether to use normalization within the attention mechanism
            num_prefix_tokens: Number of tokens at the beginning of the sequence (class tokens, etc.)
            attn_type: Type of attention module to use ('eva' or 'rope')
            proj_drop: Dropout rate for projection layers
            attn_drop: Dropout rate for attention matrix
            drop_path: Stochastic depth rate
            init_values: Initial value for LayerScale, None = no LayerScale
            act_layer: Activation layer constructor
            norm_layer: Normalization layer constructor
            attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn_cls = AttentionRope if attn_type == 'rope' else EvaAttention
        self.attn = attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer,
            scale_norm=scale_attn_inner,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class EvaBlockPostNorm(nn.Module):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            attn_type: str = 'eva',
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,  # ignore for post-norm
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """ Initialize the post-norm EVA transformer block.

        Args:
          dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias terms in query, key, value projections
            qkv_fused: Whether to use a single projection for query, key, value
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            swiglu_mlp: Whether to use SwiGLU activation in the MLP
            scale_mlp: Whether to use normalization in the MLP
            scale_attn_inner: Whether to use normalization within the attention mechanism
            num_prefix_tokens: Number of tokens at the beginning of the sequence (class tokens, etc.)
            attn_type: Type of attention module to use ('eva' or 'rope')
            proj_drop: Dropout rate for projection layers
            attn_drop: Dropout rate for attention matrix
            drop_path: Stochastic depth rate
            init_values: Initial value for LayerScale, None = no LayerScale (NOTE: ignored for post-norm block)
            act_layer: Activation layer constructor
            norm_layer: Normalization layer constructor
            attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
        """
        super().__init__()
        attn_cls = AttentionRope if attn_type == 'rope' else EvaAttention
        self.attn = attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer,
            scale_norm=scale_attn_inner,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
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
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            attn_type: str = 'eva',
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            num_reg_tokens: int = 0,
            no_embed_class: bool = False,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            rope_mixed_mode: bool = False,
            rope_grid_offset: float = 0.,
            rope_grid_indexing: str = 'ij',
            rope_temperature: float = 10000.,
            use_post_norm: bool = False,
            use_pre_transformer_norm: bool = False,
            use_post_transformer_norm: Optional[bool] = None,
            use_fc_norm: Optional[bool] = None,
            attn_pool_num_heads: Optional[int] = None,
            attn_pool_mlp_ratio: Optional[float] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
    ):
        """Initialize the EVA Vision Transformer model.

        Args:
            img_size: Input image size (single int for square, or tuple for rectangular)
            patch_size: Patch size to divide image into tokens (single int for square, or tuple)
            in_chans: Number of input image channels
            num_classes: Number of classes (output dim) for classification head (final projection), 0 for pass-through
            global_pool: Type of global pooling for final sequence ('avg', 'token', 'map', etc.)
            embed_dim: Embedding dimension for tokens
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            qkv_bias: Enable bias for query, key, value projections
            qkv_fused: Use a single projection for query, key, value
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            swiglu_mlp: Use SwiGLU activation in MLP
            scale_mlp: Apply scaling normalization in MLP (normformer style)
            scale_attn_inner: Apply scaling normalization inside attention
            attn_type: Type of attention module to use
            drop_rate: Dropout rate after final projection and pooling
            pos_drop_rate: Dropout rate for positional embeddings
            patch_drop_rate: Rate of dropping patches during training
            proj_drop_rate: Dropout rate for projections
            attn_drop_rate: Dropout rate for attention
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer constructor
            init_values: Initial layer-scale values
            class_token: Use class token
            num_reg_tokens: Number of additional learnable 'register' tokens to add to the sequence
            no_embed_class: Don't include position embeddings for class (or reg) tokens
            use_abs_pos_emb: Use absolute (learned) positional embeddings
            use_rot_pos_emb: Use rotary position embeddings
            rope_mixed_mode: Use mixed mode ROPE with per-layer learnable frequencies
            rope_grid_offset: Offset for rotary position embedding grid
            rope_grid_indexing: Indexing mode for rotary position embeddings ('ij' or 'xy')
            rope_temperature: Temperature parameter for ROPE frequency computation
            use_post_norm: Use post-norm transformer block type
            use_pre_transformer_norm: Use normalization layer before transformer blocks
            use_post_transformer_norm: Use normalization layer after transformer blocks
            use_fc_norm: Use normalization layer after pooling, before final classifier
            attn_pool_num_heads: Number of heads in attention pooling
            attn_pool_mlp_ratio: MLP ratio in attention pooling
            dynamic_img_size: Support dynamic image sizes in forward pass
            dynamic_img_pad: Apply dynamic padding for irregular image sizes
            ref_feat_shape: Reference feature shape for rotary position embedding scale
            head_init_scale: Initialization scale for classification head weights
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        # resolve norm / pool usage
        activate_pre_norm = use_pre_transformer_norm
        if use_fc_norm is not None:
            activate_fc_norm = use_fc_norm  # pass through if explicit
        else:
            activate_fc_norm = global_pool == 'avg'  # default on if avg pool used
        if use_post_transformer_norm is not None:
            activate_post_norm = use_post_transformer_norm  # pass through if explicit
        else:
            activate_post_norm = not activate_fc_norm  # default on if fc_norm isn't active

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dynamic_img_pad=dynamic_img_pad,
            bias=not use_pre_transformer_norm,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None

        num_pos_tokens = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_pos_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropoutWithIndices(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            if rope_mixed_mode:
                self.rope_mixed = True
                # Mixed mode to supports depth-dependent frequencies
                self.rope = RotaryEmbeddingMixed(
                    dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    temperature=rope_temperature,
                    feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                    grid_indexing=rope_grid_indexing,
                )
            else:
                self.rope_mixed = False
                self.rope = RotaryEmbeddingCat(
                    dim=embed_dim // num_heads,
                    temperature=rope_temperature,
                    in_pixels=False,
                    feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                    ref_feat_shape=ref_feat_shape,
                    grid_offset=rope_grid_offset,
                    grid_indexing=rope_grid_indexing,
                )
        else:
            self.rope_mixed = False
            self.rope = None

        self.norm_pre = norm_layer(embed_dim) if activate_pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                attn_type=attn_type,
                num_prefix_tokens=self.num_prefix_tokens,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        self.norm =  norm_layer(embed_dim) if activate_post_norm else nn.Identity()

        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=attn_pool_num_heads or num_heads,
                mlp_ratio=attn_pool_mlp_ratio or mlp_ratio,
                norm_layer=norm_layer,
                act_layer=nn.GELU,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if activate_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self) -> None:
        """Fix initialization weights by rescaling based on layer depth."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for Linear layers.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Parameters to exclude from weight decay."""
        nwd = {'pos_embed', 'cls_token'}
        if (rope := getattr(self, "rope", None)) and hasattr(rope, "no_weight_decay"):
            return nwd | {f"rope.{p}" for p in rope.no_weight_decay()}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing."""
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        """Create layer groupings for optimization."""
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of output classes.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update the input image resolution and patch size.

        Args:
            img_size: New input resolution, if None current resolution is used.
            patch_size: New patch size, if None existing patch size is used.
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)

        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=self.patch_embed.grid_size,
                    old_size=prev_grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    verbose=True,
                ))

        if self.rope is not None:
            self.rope.update_feat_shape(self.patch_embed.grid_size)

    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                prev_grid_size = self.patch_embed.grid_size
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=(H, W),
                    old_size=prev_grid_size,
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # position embedding does not overlap with class / reg token
            if pos_embed is not None:
                x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # pos_embed has entry for class / reg token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            if pos_embed is not None:
                x = x + pos_embed

        x = self.pos_drop(x)

        # apply patch dropout to patches and rotary position embedding
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
                # After applying keep indices to rope embeds, batch dim is added
                if getattr(self, 'rope_mixed', False):
                    # B, D, nH, N, dim -> D, B, nH, N, dim. For consistent iteration over depth at index 0.
                    rot_pos_embed = rot_pos_embed.transpose(0, 1)
                else:
                    # B, N, dim -> B, 1, N, dim.  Need head dim singleton for correct dim alignment in axial mode.
                    rot_pos_embed = rot_pos_embed.unsqueeze(1)

        return x, rot_pos_embed

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.
        Args:
            x: Input image tensor
            indices: Take last n blocks if an int, if is a sequence, select by matching indices
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format for EVA-ViT features must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]

        # Handle depth-dependent embeddings for mixed mode
        if getattr(self, 'rope_mixed', False) and rot_pos_embed is not None:
            for i, blk in enumerate(blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, rope=rot_pos_embed[i])
                else:
                    x = blk(x, rope=rot_pos_embed[i])
                if i in take_indices:
                    intermediates.append(self.norm(x) if norm else x)
        else:
            for i, blk in enumerate(blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, rope=rot_pos_embed)
                else:
                    x = blk(x, rope=rot_pos_embed)
                if i in take_indices:
                    intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.attn_pool = None
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)

        if getattr(self, 'rope_mixed', False) and rot_pos_embed is not None:
            # Handle depth-dependent embeddings for mixed mode
            # pos embed has shape (depth, num_heads, H*W, dim) or (depth, batch_size, num_heads, H*W, dim)
            for i, blk in enumerate(self.blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, rope=rot_pos_embed[i])
                else:
                    x = blk(x, rope=rot_pos_embed[i])
        else:
            # Standard path for non-mixed mode
            for blk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, rope=rot_pos_embed)
                else:
                    x = blk(x, rope=rot_pos_embed)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return pre-logits if True.

        Returns:
            Output tensor.
        """
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _convert_pe(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    """Convert Perception Encoder weights.

    Args:
        state_dict: State dictionary to convert.
        model: Target model instance.
        prefix: Prefix to strip from keys.

    Returns:
        Converted state dictionary.
    """
    state_dict = state_dict.get('model', state_dict)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('ls_1.gamma', 'gamma_1'),
        ('ls_2.gamma', 'gamma_2'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if not k.startswith(prefix):
                continue
            k = k[len_prefix:]

        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k.startswith('attn_pool'):
            k = k.replace('attn_pool.attn', 'attn_pool')
            k = k.replace('attn_pool.layernorm', 'attn_pool.norm')
            k = k.replace('attn_pool.probe', 'attn_pool.latent')
            if k.startswith('attn_pool.qkv'):
                dim = v.shape[0] // 3
                if k.endswith('weight'):
                    out_dict['attn_pool.q.weight'] = v[:dim]
                    out_dict['attn_pool.kv.weight'] = v[dim:]
                elif k.endswith('bias'):
                    out_dict['attn_pool.q.bias'] = v[:dim]
                    out_dict['attn_pool.kv.bias'] = v[dim:]
                continue
        elif k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
        out_dict[k] = v

    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert patch embedding weight from manual patchify + linear proj to conv.

    Args:
        state_dict: Checkpoint state dictionary.
        model: Target model instance.
        interpolation: Interpolation method for resizing.
        antialias: Whether to use antialiasing when resizing.

    Returns:
        Filtered state dictionary.
    """
    out_dict = {}
    # Standard EVA checkpoint processing
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)

    # Loading Meta PE (Perception Encoder) weights
    if 'visual.conv1.weight' in state_dict:
        return _convert_pe(state_dict, model)
    elif 'conv1.weight' in state_dict:
        return _convert_pe(state_dict, model, prefix='')

    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if not k.startswith(prefix):
                continue
            k = k[len_prefix:]

        if 'rope' in k and not k == 'rope.freqs':
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict


def _create_eva(variant: str, pretrained: bool = False, **kwargs) -> Eva:
    """Create an EVA model.

    Args:
        variant: Model variant name.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        Instantiated Eva model.
    """
    # Check if we should use NaFlexVit implementation
    use_naflex = kwargs.pop('use_naflex', None)
    _USE_NAFLEX_DEFAULT = os.environ.get('TIMM_USE_NAFLEX', '0') == '1'
    if use_naflex is None:
        use_naflex = _USE_NAFLEX_DEFAULT
    if use_naflex:
        # Import here to avoid circular imports
        from .naflexvit import _create_naflexvit_from_eva
        return _create_naflexvit_from_eva(variant, pretrained, **kwargs)

    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        Eva, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    """Generate default configuration for EVA models.

    Args:
        url: Model weights URL.
        **kwargs: Additional configuration parameters.

    Returns:
        Model configuration dictionary.
    """
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': OPENAI_CLIP_MEAN, 'std': OPENAI_CLIP_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'mit', **kwargs
    }


def _pe_cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    """Generate default configuration for Perception Encoder models.

    Args:
        url: Model weights URL.
        **kwargs: Additional configuration parameters.

    Returns:
        Model configuration dictionary.
    """
    return {
        'url': url,
        'num_classes': 0, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'custom', **kwargs
    }


default_cfgs = generate_default_cfgs({

    # EVA 01 CLIP fine-tuned on imagenet-1k
    'eva_giant_patch14_224.clip_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_clip_vis_enc_sz224_ftcls_89p1.pt',
        hf_hub_id='timm/',
    ),
    'eva_giant_patch14_336.clip_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_clip_vis_enc_sz336_ftcls_89p4.pt',
        hf_hub_id='timm/',
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),

    # MIM EVA 01 pretrain, ft on in22k -> in1k
    'eva_giant_patch14_336.m30m_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_21k_1k_336px_psz14_ema_89p6.pt',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),
    'eva_giant_patch14_560.m30m_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_21k_1k_560px_psz14_ema_89p7.pt',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 560, 560), crop_pct=1.0, crop_mode='squash'),

    # in22k or m38m MIM pretrain w/ intermediate in22k fine-tune and final in1k fine-tune
    'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k_to_in1k/eva02_B_pt_in21k_medft_in21k_ft_in1k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash',
    ),
    'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k_to_in1k/eva02_L_pt_in21k_medft_in21k_ft_in1k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash',
    ),
    'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k': _cfg(
        hf_hub_id='timm/',
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k_to_in1k/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt',
        input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash',
    ),

    # in22k or m3m MIM pretrain w/ in1k fine-tune
    'eva02_tiny_patch14_336.mim_in22k_ft_in1k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 336, 336), crop_pct=1.0,
    ),
    'eva02_small_patch14_336.mim_in22k_ft_in1k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 336, 336), crop_pct=1.0,
    ),
    'eva02_base_patch14_448.mim_in22k_ft_in1k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_B_pt_in21k_ft_in1k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 448, 448), crop_pct=1.0,
    ),
    'eva02_large_patch14_448.mim_in22k_ft_in1k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_L_pt_in21k_ft_in1k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 448, 448), crop_pct=1.0,
    ),
    'eva02_large_patch14_448.mim_m38m_ft_in1k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_L_pt_m38m_ft_in1k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 448, 448), crop_pct=1.0,
    ),

    # in22k or m3m MIM pretrain w/ in22k fine-tune
    'eva02_base_patch14_448.mim_in22k_ft_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k/eva02_B_pt_in21k_medft_in21k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash', num_classes=21841,
    ),
    'eva02_large_patch14_448.mim_in22k_ft_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k/eva02_L_pt_in21k_medft_in21k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash', num_classes=21841,
    ),
    'eva02_large_patch14_448.mim_m38m_ft_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k/eva02_L_pt_m38m_medft_in21k_p14.pt',
        hf_hub_id='timm/',
        input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash', num_classes=21841,
    ),

    # in22k or m38m MIM pretrain
    'eva02_tiny_patch14_224.mim_in22k': _cfg(
        # hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_Ti_pt_in21k_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_small_patch14_224.mim_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_S_pt_in21k_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_base_patch14_224.mim_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_B_pt_in21k_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_large_patch14_224.mim_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_in21k_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_large_patch14_224.mim_m38m': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_m38m_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),

    # EVA01 and EVA02 CLIP image towers
    'eva_giant_patch14_clip_224.laion400m': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA01_CLIP_g_14_plus_psz14_s11B.pt',
        # hf_hub_id='timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k',  # float16 weights
        # hf_hub_filename='open_clip_pytorch_model.bin',
        hf_hub_id='timm/',
        num_classes=1024,
    ),
    'eva_giant_patch14_clip_224.merged2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA01_CLIP_g_14_plus_psz14_s11B.pt',
        # hf_hub_id='timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k',  # float16 weights
        # hf_hub_filename='open_clip_pytorch_model.bin',
        hf_hub_id='timm/',
        num_classes=1024,
    ),
    'eva02_base_patch16_clip_224.merged2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
        # hf_hub_id='timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k',  # float16 weights
        # hf_hub_filename='open_clip_pytorch_model.bin',
        hf_hub_id='timm/',
        num_classes=512,
    ),
    'eva02_large_patch14_clip_224.merged2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
        # hf_hub_id='timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k',  # float16 weights
        # hf_hub_filename='open_clip_pytorch_model.bin',
        hf_hub_id='timm/',
        num_classes=768,
    ),
    'eva02_large_patch14_clip_336.merged2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
        # hf_hub_id='timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k',  # float16 weights
        # hf_hub_filename='open_clip_pytorch_model.bin',
        hf_hub_id='timm/',
        input_size=(3, 336, 336), crop_pct=1.0,
        num_classes=768,
    ),
    'eva02_enormous_patch14_clip_224.laion2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_E_psz14_plus_s9B.pt',
        # hf_hub_id='timm/eva02_enormous_patch14_clip_224.laion2b_s4b_b115k',  # float16 weights
        # hf_hub_filename='open_clip_pytorch_model.bin',
        hf_hub_id='timm/',
        num_classes=1024,
    ),
    'eva02_enormous_patch14_clip_224.laion2b_plus': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_E_psz14_plus_s9B.pt',
        # hf_hub_id='timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k',  # bfloat16 weights
        # hf_hub_filename='open_clip_pytorch_model.bin',
        hf_hub_id='timm/',
        num_classes=1024,
    ),
    'eva02_enormous_patch14_clip_224.pretrain': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_E_psz14.pt',
        num_classes=0,
    ),

    'vit_medium_patch16_rope_reg1_gap_256.sbb_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    ),
    'vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    ),
    'vit_betwixt_patch16_rope_reg4_gap_256.sbb_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95,
    ),
    'vit_base_patch16_rope_reg1_gap_256.sbb_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    ),

    # Perception Encoder weights
    'vit_pe_core_tiny_patch16_384.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Core-T16-384',
        #hf_hub_filename='PE-Core-T16-384.pt',
        input_size=(3, 384, 384),
        num_classes=512,  # output proj dim
    ),
    'vit_pe_core_small_patch16_384.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Core-S16-384',
        #hf_hub_filename='PE-Core-S16-384.pt',
        input_size=(3, 384, 384),
        num_classes=512,  # output proj dim
    ),
    'vit_pe_core_base_patch16_224.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Core-B16-224',
        #hf_hub_filename='PE-Core-B16-224.pt',
        input_size=(3, 224, 224),
        num_classes=1024,  # output proj dim
    ),
    'vit_pe_core_large_patch14_336.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Core-L14-336',
        #hf_hub_filename='PE-Core-L14-336.pt',
        input_size=(3, 336, 336),
        num_classes=1024,  # output proj dim
    ),
    'vit_pe_core_gigantic_patch14_448.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Core-G14-448',
        #hf_hub_filename='PE-Core-G14-448.pt',
        input_size=(3, 448, 448),
        num_classes=1280,  # output proj dim
    ),

    'vit_pe_lang_large_patch14_448.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Lang-L14-448',
        #hf_hub_filename='PE-Lang-L14-448.pt',
        input_size=(3, 448, 448),
        num_classes=0,
    ),
    'vit_pe_lang_large_patch14_448.fb_tiling': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Lang-L14-448-Tiling',
        #hf_hub_filename='PE-Lang-L14-448-Tiling.pt',
        input_size=(3, 448, 448),
        num_classes=0,
    ),
    'vit_pe_lang_gigantic_patch14_448.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Lang-G14-448',
        #hf_hub_filename='PE-Lang-G14-448.pt',
        input_size=(3, 448, 448),
        num_classes=0,
    ),
    'vit_pe_lang_gigantic_patch14_448.fb_tiling': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Lang-G14-448-Tiling',
        #hf_hub_filename='PE-Lang-G14-448-Tiling.pt',
        input_size=(3, 448, 448),
        num_classes=0,
    ),

    'vit_pe_spatial_tiny_patch16_512.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Spatial-T16-512',
        #hf_hub_filename='PE-Spatial-T16-512.pt',
        input_size=(3, 512, 512),
        num_classes=0,
    ),
    'vit_pe_spatial_small_patch16_512.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Spatial-S16-512',
        #hf_hub_filename='PE-Spatial-S16-512.pt',
        input_size=(3, 512, 512),
        num_classes=0,
    ),
    'vit_pe_spatial_base_patch16_512.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Spatial-B16-512',
        #hf_hub_filename='PE-Spatial-B16-512.pt',
        input_size=(3, 512, 512),
        num_classes=0,
    ),
    'vit_pe_spatial_large_patch14_448.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Spatial-L14-448',
        #hf_hub_filename='PE-Spatial-L14-448.pt',
        input_size=(3, 448, 448),
        num_classes=0,
    ),
    'vit_pe_spatial_gigantic_patch14_448.fb': _pe_cfg(
        hf_hub_id='timm/',
        #hf_hub_id='facebook/PE-Spatial-G14-448',
        #hf_hub_filename='PE-Spatial-G14-448.pt',
        input_size=(3, 448, 448),
        num_classes=0,
    ),

    # RoPE-ViT models from Naver
    'vit_small_patch16_rope_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_base_patch16_rope_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_large_patch16_rope_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_small_patch16_rope_mixed_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_base_patch16_rope_mixed_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_large_patch16_rope_mixed_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_small_patch16_rope_ape_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_base_patch16_rope_ape_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_large_patch16_rope_ape_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_small_patch16_rope_mixed_ape_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_base_patch16_rope_mixed_ape_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
    'vit_large_patch16_rope_mixed_ape_224.naver_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        license='apache-2.0',
    ),
})


@register_model
def eva_giant_patch14_224(pretrained: bool = False, **kwargs) -> Eva:
    """EVA-g model https://arxiv.org/abs/2211.07636"""
    model_args = dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408)
    model = _create_eva('eva_giant_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva_giant_patch14_336(pretrained: bool = False, **kwargs) -> Eva:
    """EVA-g model https://arxiv.org/abs/2211.07636"""
    model_args = dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408)
    model = _create_eva('eva_giant_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva_giant_patch14_560(pretrained: bool = False, **kwargs) -> Eva:
    """EVA-g model https://arxiv.org/abs/2211.07636"""
    model_args = dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408)
    model = _create_eva('eva_giant_patch14_560', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_tiny_patch14_224(pretrained: bool = False, **kwargs) -> Eva:
    """EVA02 Tiny https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('eva02_tiny_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_small_patch14_224(pretrained: bool = False, **kwargs) -> Eva:
    """EVA02 Small https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('eva02_small_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_base_patch14_224(pretrained: bool = False, **kwargs) -> Eva:
    """EVA02 Base https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('eva02_base_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_large_patch14_224(pretrained: bool = False, **kwargs) -> Eva:
    """EVA02 Large https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('eva02_large_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_tiny_patch14_336(pretrained: bool = False, **kwargs) -> Eva:
    """EVA02 Tiny https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=336,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('eva02_tiny_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_small_patch14_336(pretrained: bool = False, **kwargs) -> Eva:
    """EVA02 Small https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=336,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('eva02_small_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_base_patch14_448(pretrained: bool = False, **kwargs) -> Eva:
    """EVA02 Base https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=448,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('eva02_base_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_large_patch14_448(pretrained: bool = False, **kwargs) -> Eva:
    """EVA02 Large https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=448,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('eva02_large_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva_giant_patch14_clip_224(pretrained: bool = False, **kwargs) -> Eva:
    """EVA-g CLIP model (only difference from non-CLIP is the pooling)"""
    model_args = dict(
        patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408,
        global_pool=kwargs.pop('global_pool', 'token'))
    model = _create_eva('eva_giant_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_base_patch16_clip_224(pretrained: bool = False, **kwargs) -> Eva:
    """An EVA-CLIP specific variant that adds additional attn scale layer-norm to eva02_base"""
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
    )
    model = _create_eva('eva02_base_patch16_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_large_patch14_clip_224(pretrained: bool = False, **kwargs) -> Eva:
    """An EVA-CLIP specific variant that adds additional attn scale layer-norm to eva02_large"""
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
    )
    model = _create_eva('eva02_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_large_patch14_clip_336(pretrained: bool = False, **kwargs) -> Eva:
    """An EVA-CLIP specific variant that adds additional attn scale layer-norm to eva02_large"""
    model_args = dict(
        img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
    )
    model = _create_eva('eva02_large_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva02_enormous_patch14_clip_224(pretrained: bool = False, **kwargs) -> Eva:
    """An EVA-CLIP specific variant that uses residual post-norm in blocks"""
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1792,
        depth=64,
        num_heads=16,
        mlp_ratio=15360 / 1792,
        use_post_norm=True,
        global_pool=kwargs.pop('global_pool', 'token'),
    )
    model = _create_eva('eva02_enormous_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_medium_patch16_rope_reg1_gap_256(pretrained: bool = False, **kwargs) -> Eva:
    """timm SBB ViT with ROPE"""
    model_args = dict(
        img_size=256,
        patch_size=16,
        embed_dim=512,
        depth=12,
        num_heads=8,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=1,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('vit_medium_patch16_rope_reg1_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_mediumd_patch16_rope_reg1_gap_256(pretrained: bool = False, **kwargs) -> Eva:
    """timm SBB ViT with ROPE"""
    model_args = dict(
        img_size=256,
        patch_size=16,
        embed_dim=512,
        depth=20,
        num_heads=8,
        qkv_fused=True,
        qkv_bias=False,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=1,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('vit_mediumd_patch16_rope_reg1_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_betwixt_patch16_rope_reg4_gap_256(pretrained: bool = False, **kwargs) -> Eva:
    """timm SBB ViT with ROPE"""
    model_args = dict(
        img_size=256,
        patch_size=16,
        embed_dim=640,
        depth=12,
        num_heads=10,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=4,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('vit_betwixt_patch16_rope_reg4_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_rope_reg1_gap_256(pretrained: bool = False, **kwargs) -> Eva:
    """timm SBB ViT with ROPE"""
    model_args = dict(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=1,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = _create_eva('vit_base_patch16_rope_reg1_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_pe_core_tiny_patch16_384(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        global_pool='map',
        attn_type='rope',
        use_pre_transformer_norm=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(24, 24),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.,
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True
    )
    return _create_eva('vit_pe_core_tiny_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))



@register_model
def vit_pe_core_small_patch16_384(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        global_pool='map',
        attn_type='rope',
        use_pre_transformer_norm=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(24, 24),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.,
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True
    )
    return _create_eva('vit_pe_core_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_core_base_patch16_224(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        global_pool='map',
        attn_type='rope',
        use_pre_transformer_norm=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.,
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True
    )
    return _create_eva('vit_pe_core_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_core_large_patch14_336(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        global_pool='map',
        attn_type='rope',
        use_pre_transformer_norm=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(24, 24),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.,
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True,
    )
    return _create_eva('vit_pe_core_large_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_core_gigantic_patch14_448(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=14,
        embed_dim=1536,
        depth=50,
        num_heads=16,
        mlp_ratio=8960 / 1536,
        global_pool='map',
        attn_type='rope',
        class_token=False,
        use_pre_transformer_norm=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(32, 32),
        rope_grid_indexing='xy',
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.,
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True,
    )
    return _create_eva('vit_pe_core_gigantic_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_lang_large_patch14_448(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=14,
        embed_dim=1024,
        depth=23,
        num_heads=16,
        mlp_ratio=4.0,
        attn_type='rope',
        class_token=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(32, 32),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        use_pre_transformer_norm=True,
        use_post_transformer_norm=False,
        use_fc_norm=False,  # explicitly disable
        init_values=0.1,
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True,
    )
    return _create_eva('vit_pe_lang_large_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_lang_gigantic_patch14_448(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=14,
        embed_dim=1536,
        depth=47,
        num_heads=16,
        mlp_ratio=8960 / 1536,
        attn_type='rope',
        class_token=False,
        use_rot_pos_emb=True,
        ref_feat_shape=(32, 32),
        rope_grid_indexing='xy',
        use_pre_transformer_norm=True,
        use_post_transformer_norm=False,
        use_fc_norm=False,  # explicitly disable
        init_values=0.1,
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True,
    )
    return _create_eva('vit_pe_lang_gigantic_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_spatial_tiny_patch16_512(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        attn_type='rope',
        use_pre_transformer_norm=True,
        use_post_transformer_norm=False,
        use_fc_norm=False,  # explicitly disable
        use_rot_pos_emb=True,
        ref_feat_shape=(32, 32),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True
    )
    return _create_eva('vit_pe_spatial_tiny_patch16_512', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_spatial_small_patch16_512(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        attn_type='rope',
        use_pre_transformer_norm=True,
        use_post_transformer_norm=False,
        use_fc_norm=False,  # explicitly disable
        use_rot_pos_emb=True,
        ref_feat_shape=(32, 32),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True
    )
    return _create_eva('vit_pe_spatial_small_patch16_512', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_spatial_base_patch16_512(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        attn_type='rope',
        use_pre_transformer_norm=True,
        use_post_transformer_norm=False,
        use_fc_norm=False,  # explicitly disable
        use_rot_pos_emb=True,
        ref_feat_shape=(32, 32),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True
    )
    return _create_eva('vit_pe_spatial_base_patch16_512', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_spatial_large_patch14_448(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_type='rope',
        use_pre_transformer_norm=True,
        use_post_transformer_norm=False,
        use_fc_norm=False,  # explicitly disable
        use_rot_pos_emb=True,
        ref_feat_shape=(32, 32),
        rope_grid_offset=1.,
        rope_grid_indexing='xy',
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True,
    )
    return _create_eva('vit_pe_spatial_large_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_spatial_gigantic_patch14_448(pretrained: bool = False, **kwargs) -> Eva:
    """Perception Encoder (PE) ViT from Meta (https://arxiv.org/abs/2504.13181)"""
    model_args = dict(
        patch_size=14,
        embed_dim=1536,
        depth=50,
        num_heads=16,
        mlp_ratio=8960 / 1536,
        attn_type='rope',
        class_token=False,
        use_rot_pos_emb=True,
        ref_feat_shape=(32, 32),
        rope_grid_indexing='xy',
        use_pre_transformer_norm=True,
        use_post_transformer_norm=False,
        use_fc_norm=False,  # explicitly disable
        init_values=0.1,
        norm_layer=partial(LayerNorm, eps=1e-5),
        #dynamic_img_size=True,
    )
    return _create_eva('vit_pe_spatial_gigantic_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))


# RoPE-ViT models from https://github.com/naver-ai/rope-vit
@register_model
def vit_small_patch16_rope_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Axial ViT-S/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=100.0,
    )
    model = _create_eva('vit_small_patch16_rope_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_rope_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Axial ViT-B/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        attn_type='rope',
        use_fc_norm=False,
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=100.0,
    )
    model = _create_eva('vit_base_patch16_rope_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_rope_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Axial ViT-L/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=100.0,
    )
    model = _create_eva('vit_large_patch16_rope_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch16_rope_mixed_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Mixed ViT-S/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=10.0,
        rope_mixed_mode=True,
    )
    model = _create_eva('vit_small_patch16_rope_mixed_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_rope_mixed_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Mixed ViT-B/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        attn_type='rope',
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=10.0,
        rope_mixed_mode=True,
    )
    model = _create_eva('vit_base_patch16_rope_mixed_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_rope_mixed_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Mixed ViT-L/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=10.0,
        rope_mixed_mode=True,
    )
    model = _create_eva('vit_large_patch16_rope_mixed_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# APE variants (with absolute position embeddings)
@register_model
def vit_small_patch16_rope_ape_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Axial + APE ViT-S/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        no_embed_class=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=100.0,
    )
    model = _create_eva('vit_small_patch16_rope_ape_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_rope_ape_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Axial + APE ViT-B/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        no_embed_class=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=100.0,
    )

    model = _create_eva('vit_base_patch16_rope_ape_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_rope_ape_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Axial + APE ViT-L/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        no_embed_class=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=100.0,
    )
    
    model = _create_eva('vit_large_patch16_rope_ape_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch16_rope_mixed_ape_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Mixed + APE ViT-S/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        no_embed_class=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=10.0,
        rope_mixed_mode=True,
    )

    model = _create_eva('vit_small_patch16_rope_mixed_ape_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_rope_mixed_ape_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Mixed + APE ViT-B/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        no_embed_class=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=10.0,
        rope_mixed_mode=True,
    )
    model = _create_eva('vit_base_patch16_rope_mixed_ape_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_rope_mixed_ape_224(pretrained: bool = False, **kwargs) -> Eva:
    """RoPE-Mixed + APE ViT-L/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        attn_type='rope',
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        no_embed_class=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=10.0,
        rope_mixed_mode=True,
    )
    model = _create_eva('vit_large_patch16_rope_mixed_ape_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

