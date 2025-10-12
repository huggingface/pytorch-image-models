""" BEiT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)

Model from official source: https://github.com/microsoft/unilm/tree/master/beit

@inproceedings{beit,
title={{BEiT}: {BERT} Pre-Training of Image Transformers},
author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=p-BhZSz59o4}
}

BEiT-v2 from https://github.com/microsoft/unilm/tree/master/beit2

@article{beitv2,
title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
year={2022},
eprint={2208.06366},
archivePrefix={arXiv},
primaryClass={cs.CV}
}

At this point only the 1k fine-tuned classification weights and model configs have been added,
see original source above for pre-training models and procedure.

Modifications by / Copyright 2021 Ross Wightman, original copyrights below
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    PatchEmbed,
    Mlp,
    SwiGLU,
    LayerNorm,
    DropPath,
    calculate_drop_path_rates,
    trunc_normal_,
    use_fused_attn,
    resample_patch_embed,
    resample_abs_pos_embed,
    resize_rel_pos_bias_table,
    ndgrid,
)

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model

__all__ = ['Beit']


def gen_relative_position_index(window_size: Tuple[int, int], device=None) -> torch.Tensor:
    """Generate relative position index for window-based attention.

    Creates a lookup table for relative position indices between all pairs of positions
    within a window, including special handling for cls token interactions.

    Args:
        window_size: Height and width of the attention window.

    Returns:
        Relative position index tensor of shape (window_area+1, window_area+1)
        where +1 accounts for the cls token.
    """
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(ndgrid(
        torch.arange(window_size[0], device=device, dtype=torch.long),
        torch.arange(window_size[1], device=device, dtype=torch.long),
    ))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(size=(window_area + 1,) * 2, device=device, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class Attention(nn.Module):
    """Multi-head attention module with optional relative position bias.

    Implements multi-head self-attention with support for relative position bias
    and fused attention operations. Can use either standard or custom head dimensions.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            window_size: Optional[Tuple[int, int]] = None,
            attn_head_dim: Optional[int] = None,
            device=None,
            dtype=None,
    ):
        """Initialize attention module.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            qkv_bias: If True, add learnable bias to query, key, value projections.
            qkv_bias_separate: If True, use separate bias for q, k, v projections.
            attn_drop: Dropout rate for attention weights.
            proj_drop: Dropout rate for output projection.
            window_size: Window size for relative position bias. If None, no relative position bias.
            attn_head_dim: Dimension per attention head. If None, uses dim // num_heads.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.qkv_bias_separate = qkv_bias_separate

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False, **dd)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim, **dd))
            self.register_buffer('k_bias', torch.zeros(all_head_dim, **dd), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim, **dd))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads, **dd))  # 2*Wh-1 * 2*Ww-1, nH
            self.register_buffer(
                "relative_position_index",
                gen_relative_position_index(window_size, device=device),
                persistent=False,
            )
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        """Get relative position bias for the attention window.

        Returns:
            Relative position bias tensor of shape (1, num_heads, window_area+1, window_area+1).
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor, shared_rel_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of attention module.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, dim).
            shared_rel_pos_bias: Optional shared relative position bias from parent module.

        Returns:
            Output tensor of shape (batch_size, num_tokens, dim).
        """
        B, N, C = x.shape

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

        if self.fused_attn:
            rel_pos_bias = None
            if self.relative_position_bias_table is not None:
                rel_pos_bias = self._get_rel_pos_bias()
                if shared_rel_pos_bias is not None:
                    rel_pos_bias = rel_pos_bias + shared_rel_pos_bias
            elif shared_rel_pos_bias is not None:
                rel_pos_bias = shared_rel_pos_bias

            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=rel_pos_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                attn = attn + self._get_rel_pos_bias()
            if shared_rel_pos_bias is not None:
                attn = attn + shared_rel_pos_bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP.

    Standard transformer block consisting of multi-head self-attention and MLP
    with residual connections and layer normalization. Supports layer scale and
    stochastic depth regularization.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            mlp_ratio: float = 4.,
            scale_mlp: bool = False,
            swiglu_mlp: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            window_size: Optional[Tuple[int, int]] = None,
            attn_head_dim: Optional[int] = None,
            device=None,
            dtype=None,
    ):
        """Initialize transformer block.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            qkv_bias: If True, add learnable bias to query, key, value projections.
            mlp_ratio: Ratio of MLP hidden dimension to input dimension.
            scale_mlp: If True, apply layer normalization in MLP.
            swiglu_mlp: If True, use SwiGLU activation in MLP.
            proj_drop: Dropout rate for projections.
            attn_drop: Dropout rate for attention.
            drop_path: Drop path rate for stochastic depth.
            init_values: Initial values for layer scale. If None, no layer scale.
            act_layer: Activation function class.
            norm_layer: Normalization layer class.
            window_size: Window size for relative position bias in attention.
            attn_head_dim: Dimension per attention head.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.norm1 = norm_layer(dim, **dd)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
            **dd,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        if swiglu_mlp:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
                **dd,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
                **dd,
            )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if init_values:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim, **dd))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim, **dd))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor, shared_rel_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, dim).
            shared_rel_pos_bias: Optional shared relative position bias.

        Returns:
            Output tensor of shape (batch_size, num_tokens, dim).
        """
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Module):
    """Relative position bias module for window-based attention.

    Generates learnable relative position biases for all pairs of positions
    within a window, including special handling for cls token.
    """

    def __init__(self, window_size: Tuple[int, int], num_heads: int, device=None, dtype=None):
        """Initialize relative position bias module.

        Args:
            window_size: Height and width of the attention window.
            num_heads: Number of attention heads.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads, **dd))
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.register_buffer("relative_position_index", gen_relative_position_index(window_size))

    def forward(self) -> torch.Tensor:
        """Generate relative position bias.

        Returns:
            Relative position bias tensor of shape (num_heads, window_area+1, window_area+1).
        """
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area + 1, self.window_area + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class Beit(nn.Module):
    """BEiT: BERT Pre-Training of Image Transformers.

    Vision Transformer model with support for relative position bias and
    shared relative position bias across layers. Implements both BEiT v1 and v2
    architectures with flexible configuration options.
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
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Type[nn.Module] = LayerNorm,
            init_values: Optional[float] = None,
            use_abs_pos_emb: bool = True,
            use_rel_pos_bias: bool = False,
            use_shared_rel_pos_bias: bool = False,
            head_init_scale: float = 0.001,
            device=None,
            dtype=None,
    ):
        """Initialize BEiT model.

        Args:
            img_size: Input image size.
            patch_size: Patch size for patch embedding.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling ('avg' or '').
            embed_dim: Embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            qkv_bias: If True, add learnable bias to query, key, value projections.
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
            swiglu_mlp: If True, use SwiGLU activation in MLP.
            scale_mlp: If True, apply layer normalization in MLP.
            drop_rate: Dropout rate.
            pos_drop_rate: Dropout rate for position embeddings.
            proj_drop_rate: Dropout rate for projections.
            attn_drop_rate: Dropout rate for attention.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer class.
            init_values: Initial values for layer scale.
            use_abs_pos_emb: If True, use absolute position embeddings.
            use_rel_pos_bias: If True, use relative position bias in attention.
            use_shared_rel_pos_bias: If True, share relative position bias across layers.
            head_init_scale: Scale factor for head initialization.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            **dd,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, **dd))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim, **dd)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.grid_size,
                num_heads=num_heads,
                **dd,
            )
        else:
            self.rel_pos_bias = None

        dpr = calculate_drop_path_rates(drop_path_rate, depth)  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                mlp_ratio=mlp_ratio,
                scale_mlp=scale_mlp,
                swiglu_mlp=swiglu_mlp,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.grid_size if use_rel_pos_bias else None,
                **dd,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim, **dd)
        self.fc_norm = norm_layer(embed_dim, **dd) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        """Fix initialization weights according to BEiT paper.

        Rescales attention and MLP weights based on layer depth to improve
        training stability.
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module):
        """Initialize model weights.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Get parameter names that should not use weight decay.

        Returns:
            Set of parameter names to exclude from weight decay.
        """
        nwd = {'pos_embed', 'cls_token'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing.

        Args:
            enable: If True, enable gradient checkpointing.
        """
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        """Create parameter group matcher for optimizer parameter groups.

        Args:
            coarse: If True, use coarse grouping.

        Returns:
            Dictionary mapping group names to regex patterns.
        """
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed|rel_pos_bias',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head.

        Returns:
            The classification head module.
        """
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        """Reset the classification head.

        Args:
            num_classes: Number of classes for new head.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        if global_pool is not None:
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
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass that returns intermediate feature maps.

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width).
            indices: Block indices to return features from. If int, returns last n blocks.
            return_prefix_tokens: If True, return both prefix and spatial tokens.
            norm: If True, apply normalization to intermediate features.
            stop_early: If True, stop at last selected intermediate.
            output_fmt: Output format ('NCHW' or 'NLC').
            intermediates_only: If True, only return intermediate features.

        Returns:
            If intermediates_only is True, returns list of intermediate tensors.
            Otherwise, returns tuple of (final_features, intermediates).
        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
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
    ) -> List[int]:
        """Prune layers not required for specified intermediate outputs.

        Args:
            indices: Indices of blocks to keep.
            prune_norm: If True, remove final normalization.
            prune_head: If True, remove classification head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Feature tensor of shape (batch_size, num_tokens, embed_dim).
        """
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            x: Feature tensor of shape (batch_size, num_tokens, embed_dim).
            pre_logits: If True, return features before final linear layer.

        Returns:
            Logits tensor of shape (batch_size, num_classes) or pre-logits.
        """
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    """Create a default configuration dictionary for BEiT models.

    Args:
        url: Model weights URL.
        **kwargs: Additional configuration parameters.

    Returns:
        Configuration dictionary.
    """
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'apache-2.0',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'beit_base_patch16_224.in22k_ft_in22k_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/'),
    'beit_base_patch16_384.in22k_ft_in22k_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_384_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_base_patch16_224.in22k_ft_in22k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth',
        hf_hub_id='timm/',
        num_classes=21841,
    ),
    'beit_large_patch16_224.in22k_ft_in22k_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/'),
    'beit_large_patch16_384.in22k_ft_in22k_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_384_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_large_patch16_512.in22k_ft_in22k_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/',
        input_size=(3, 512, 512), crop_pct=1.0,
    ),
    'beit_large_patch16_224.in22k_ft_in22k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth',
        hf_hub_id='timm/',
        num_classes=21841,
    ),

    'beitv2_base_patch16_224.in1k_ft_in22k_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21kto1k.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_base_patch16_224.in1k_ft_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft1k.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_base_patch16_224.in1k_ft_in22k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth',
        hf_hub_id='timm/',
        num_classes=21841, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224.in1k_ft_in22k_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21kto1k.pth',
        hf_hub_id='timm/',
        crop_pct=0.95, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224.in1k_ft_in1k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft1k.pth',
        hf_hub_id='timm/',
        crop_pct=0.95, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224.in1k_ft_in22k': _cfg(
        #url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth',
        hf_hub_id='timm/',
        num_classes=21841, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
})


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module, interpolation: str = 'bicubic', antialias: bool = True) -> Dict[str, torch.Tensor]:
    """Filter and process checkpoint state dict for loading.

    Handles resizing of patch embeddings, position embeddings, and relative position
    bias tables when model size differs from checkpoint.

    Args:
        state_dict: Checkpoint state dictionary.
        model: Target model to load weights into.
        interpolation: Interpolation method for resizing.
        antialias: If True, use antialiasing when resizing.

    Returns:
        Filtered state dictionary.
    """
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    # beit v2 didn't strip module

    out_dict = {}
    for k, v in state_dict.items():
        if 'relative_position_index' in k:
            continue
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
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
            num_prefix_tokens = 1
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif k.endswith('relative_position_bias_table'):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )
        out_dict[k] = v
    return out_dict


def _create_beit(variant: str, pretrained: bool = False, **kwargs) -> Beit:
    """Create a BEiT model.

    Args:
        variant: Model variant name.
        pretrained: If True, load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        BEiT model instance.
    """
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        Beit, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


@register_model
def beit_base_patch16_224(pretrained: bool = False, **kwargs) -> Beit:
    """BEiT base model @ 224x224 with patch size 16x16."""
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1)
    model = _create_beit('beit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit_base_patch16_384(pretrained: bool = False, **kwargs) -> Beit:
    """BEiT base model @ 384x384 with patch size 16x16."""
    model_args = dict(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1)
    model = _create_beit('beit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit_large_patch16_224(pretrained: bool = False, **kwargs) -> Beit:
    """BEiT large model @ 224x224 with patch size 16x16."""
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit_large_patch16_384(pretrained: bool = False, **kwargs) -> Beit:
    """BEiT large model @ 384x384 with patch size 16x16."""
    model_args = dict(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beit_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit_large_patch16_512(pretrained: bool = False, **kwargs) -> Beit:
    """BEiT large model @ 512x512 with patch size 16x16."""
    model_args = dict(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beit_large_patch16_512', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beitv2_base_patch16_224(pretrained: bool = False, **kwargs) -> Beit:
    """BEiT v2 base model @ 224x224 with patch size 16x16."""
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beitv2_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beitv2_large_patch16_224(pretrained: bool = False, **kwargs) -> Beit:
    """BEiT v2 large model @ 224x224 with patch size 16x16."""
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beitv2_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
