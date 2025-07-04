""" Swin Transformer V2
A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/abs/2111.09883

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, trunc_normal_, ClassifierHead,\
    resample_patch_embed, ndgrid, get_act_layer, LayerType
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_function
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['SwinTransformerV2']  # model_registry will add each entrypoint fn to this

_int_or_tuple_2_t = Union[int, Tuple[int, int]]


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """Partition into non-overlapping windows.

    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size: Window size (height, width).

    Returns:
        Windows tensor of shape (num_windows*B, window_size[0], window_size[1], C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int], img_size: Tuple[int, int]) -> torch.Tensor:
    """Merge windows back to feature map.

    Args:
        windows: Windows tensor of shape (num_windows * B, window_size[0], window_size[1], C).
        window_size: Window size (height, width).
        img_size: Image size (height, width).

    Returns:
        Feature map tensor of shape (B, H, W, C).
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.

    Supports both shifted and non-shifted window attention with continuous relative
    position bias and cosine attention.
    """

    def __init__(
            self,
            dim: int,
            window_size: Tuple[int, int],
            num_heads: int,
            qkv_bias: bool = True,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            pretrained_window_size: Tuple[int, int] = (0, 0),
    ) -> None:
        """Initialize window attention module.

        Args:
            dim: Number of input channels.
            window_size: The height and width of the window.
            num_heads: Number of attention heads.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qkv_bias_separate: If True, use separate bias for q, k, v projections.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
            pretrained_window_size: The height and width of the window in pre-training.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = to_2tuple(pretrained_window_size)
        self.num_heads = num_heads
        self.qkv_bias_separate = qkv_bias_separate

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) -> None:
        """Create pair-wise relative position index and coordinates table."""
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0]).to(torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1]).to(torch.float32)
        relative_coords_table = torch.stack(ndgrid(relative_coords_h, relative_coords_w))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(ndgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def set_window_size(self, window_size: Tuple[int, int]) -> None:
        """Update window size and regenerate relative position tables.

        Args:
            window_size: New window size (height, width).
        """
        window_size = to_2tuple(window_size)
        if window_size != self.window_size:
            self.window_size = window_size
            self._make_pair_wise_relative_positions()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of window attention.

        Args:
            x: Input features with shape of (num_windows*B, N, C).
            mask: Attention mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None.

        Returns:
            Output features with shape of (num_windows*B, N, C).
        """
        B_, N, C = x.shape

        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerV2Block(nn.Module):
    """Swin Transformer V2 Block.

    A standard transformer block with window attention and shifted window attention
    for modeling long-range dependencies efficiently.
    """

    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_2_t,
            num_heads: int,
            window_size: _int_or_tuple_2_t = 7,
            shift_size: _int_or_tuple_2_t = 0,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: LayerType = "gelu",
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            pretrained_window_size: _int_or_tuple_2_t = 0,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Window size.
            shift_size: Shift size for SW-MSA.
            always_partition: Always partition into full windows and shift
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            pretrained_window_size: Window size in pretraining.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.target_shift_size = to_2tuple(shift_size)  # store for later resize
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size, shift_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio
        act_layer = get_act_layer(act_layer)

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def get_attn_mask(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Generate attention mask for shifted window attention.

        Args:
            x: Input tensor for dynamic shape calculation.

        Returns:
            Attention mask or None if no shift.
        """
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            if x is None:
                img_mask = torch.zeros((1, *self.input_resolution, 1))  # 1 H W 1
            else:
                img_mask = torch.zeros((1, x.shape[1], x.shape[2], 1), dtype=x.dtype, device=x.device)  # 1 H W 1
            cnt = 0
            for h in (
                    (0, -self.window_size[0]),
                    (-self.window_size[0], -self.shift_size[0]),
                    (-self.shift_size[0], None),
            ):
                for w in (
                        (0, -self.window_size[1]),
                        (-self.window_size[1], -self.shift_size[1]),
                        (-self.shift_size[1], None),
                ):
                    img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def _calc_window_shift(
            self,
            target_window_size: _int_or_tuple_2_t,
            target_shift_size: Optional[_int_or_tuple_2_t] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate window size and shift size based on input resolution.

        Args:
            target_window_size: Target window size.
            target_shift_size: Target shift size.

        Returns:
            Tuple of (adjusted_window_size, adjusted_shift_size).
        """
        target_window_size = to_2tuple(target_window_size)
        if target_shift_size is None:
            # if passed value is None, recalculate from default window_size // 2 if it was active
            target_shift_size = self.target_shift_size
            if any(target_shift_size):
                # if there was previously a non-zero shift, recalculate based on current window_size
                target_shift_size = (target_window_size[0] // 2, target_window_size[1] // 2)
        else:
            target_shift_size = to_2tuple(target_shift_size)

        if self.always_partition:
            return target_window_size, target_shift_size

        target_window_size = to_2tuple(target_window_size)
        target_shift_size = to_2tuple(target_shift_size)
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: Tuple[int, int],
            always_partition: Optional[bool] = None,
    ) -> None:
        """Set input size and update window configuration.

        Args:
            feat_size: New feature map size.
            window_size: New window size.
            always_partition: Override always_partition setting.
        """
        # Update input resolution
        self.input_resolution = feat_size
        if always_partition is not None:
            self.always_partition = always_partition
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.set_window_size(self.window_size)
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply windowed attention with optional shift.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Output tensor of shape (B, H, W, C).
        """
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if getattr(self, 'dynamic_mask', False):
            attn_mask = self.get_attn_mask(shifted_x)
        else:
            attn_mask = self.attn_mask
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, (Hp, Wp))  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x + self.drop_path1(self.norm1(self._attn(x)))
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = x.reshape(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Merges 2x2 neighboring patches and projects to higher dimension,
    effectively downsampling the feature maps.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class SwinTransformerV2Stage(nn.Module):
    """A Swin Transformer V2 Stage.

    A single stage consisting of multiple Swin Transformer blocks with
    optional downsampling at the beginning.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: _int_or_tuple_2_t,
            depth: int,
            num_heads: int,
            window_size: _int_or_tuple_2_t,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            downsample: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            pretrained_window_size: _int_or_tuple_2_t = 0,
            output_nchw: bool = False,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            always_partition: Always partition into full windows and shift
            dynamic_mask: Create attention mask in forward based on current input size
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer type.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                always_partition=always_partition,
                dynamic_mask=dynamic_mask,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)])

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: int,
            always_partition: Optional[bool] = None,
    ) -> None:
        """Update resolution, window size and relative positions.

        Args:
            feat_size: New input (feature) resolution.
            window_size: New window size.
            always_partition: Always partition / shift the window.
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            assert isinstance(self.downsample, PatchMerging)
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stage.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Output tensor of shape (B, H', W', C').
        """
        x = self.downsample(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def _init_respostnorm(self) -> None:
        """Initialize residual post-normalization weights."""
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformerV2(nn.Module):
    """Swin Transformer V2.

    A hierarchical vision transformer using shifted windows for efficient
    self-attention computation with continuous position bias.

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 7,
            always_partition: bool = False,
            strict_img_size: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer stage (layer).
            num_heads: Number of attention heads in different layers.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Head dropout rate.
            proj_drop_rate: Projection dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            act_layer: Activation layer type.
            patch_norm: If True, add normalization after patch embedding.
            pretrained_window_sizes: Pretrained window sizes of each layer.
            output_fmt: Output tensor format if not None, otherwise output 'NHWC' by default.
        """
        super().__init__()

        self.num_classes = num_classes
        assert global_pool in ('', 'avg')
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = self.head_hidden_size = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            strict_img_size=strict_img_size,
            output_fmt='NHWC',
        )
        grid_size = self.patch_embed.grid_size

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinTransformerV2Stage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(grid_size[0] // scale, grid_size[1] // scale),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                window_size=window_size,
                always_partition=always_partition,
                dynamic_mask=not strict_img_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_sizes[i],
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=4 * scale, module=f'layers.{i}')]

        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize model weights.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
            window_size: Optional[Tuple[int, int]] = None,
            window_ratio: Optional[int] = 8,
            always_partition: Optional[bool] = None,
    ):
        """Updates the image resolution, window size, and so the pair-wise relative positions.

        Args:
            img_size (Optional[Tuple[int, int]]): New input resolution, if None current resolution is used
            patch_size (Optional[Tuple[int, int]): New patch size, if None use current patch size
            window_size (Optional[int]): New window size, if None based on new_img_size // window_div
            window_ratio (int): divisor for calculating window size from patch grid size
            always_partition: always partition / shift windows even if feat size is < window
        """
        if img_size is not None or patch_size is not None:
            self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
            grid_size = self.patch_embed.grid_size

        if window_size is None and window_ratio is not None:
            window_size = tuple([s // window_ratio for s in grid_size])

        for index, stage in enumerate(self.layers):
            stage_scale = 2 ** max(index - 1, 0)
            stage.set_input_size(
                feat_size=(grid_size[0] // stage_scale, grid_size[1] // stage_scale),
                window_size=window_size,
                always_partition=always_partition,
            )

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Get parameter names that should not use weight decay.

        Returns:
            Set of parameter names to exclude from weight decay.
        """
        nod = set()
        for n, m in self.named_modules():
            if any([kw in n for kw in ("cpb_mlp", "logit_scale")]):
                nod.add(n)
        return nod

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        """Create parameter group matcher for optimizer parameter groups.

        Args:
            coarse: If True, use coarse grouping.

        Returns:
            Dictionary mapping group names to regex patterns.
        """
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing.

        Args:
            enable: If True, enable gradient checkpointing.
        """
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head.

        Returns:
            The classification head module.
        """
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classification head.

        Args:
            num_classes: Number of classes for new head.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.layers), indices)

        # forward pass
        x = self.patch_embed(x)

        num_stages = len(self.layers)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.layers
        else:
            stages = self.layers[:max_index + 1]
        for i, stage in enumerate(stages):
            x = stage(x)
            if i in take_indices:
                if norm and i == num_stages - 1:
                    x_inter = self.norm(x)  # applying final norm last intermediate
                else:
                    x_inter = x
                x_inter = x_inter.permute(0, 3, 1, 2).contiguous()
                intermediates.append(x_inter)

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
        take_indices, max_index = feature_take_indices(len(self.layers), indices)
        self.layers = self.layers[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, H', W', C).
        """
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            x: Feature tensor of shape (B, H, W, C).
            pre_logits: If True, return features before final linear layer.

        Returns:
            Logits tensor of shape (B, num_classes) or pre-logits.
        """
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """Filter and process checkpoint state dict for loading.

    Handles resizing of patch embeddings and relative position tables
    when model size differs from checkpoint.

    Args:
        state_dict: Checkpoint state dictionary.
        model: Target model to load weights into.

    Returns:
        Filtered state dictionary.
    """
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    native_checkpoint = 'head.fc.weight' in state_dict
    out_dict = {}
    import re
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'relative_coords_table', 'attn_mask')]):
            continue  # skip buffers that should not be persistent

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-2] != H or v.shape[-1] != W:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation='bicubic',
                    antialias=True,
                    verbose=True,
                )

        if not native_checkpoint:
            # skip layer remapping for updated checkpoints
            k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
            k = k.replace('head.', 'head.fc.')
        out_dict[k] = v

    return out_dict


def _create_swin_transformer_v2(variant: str, pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Create a Swin Transformer V2 model.

    Args:
        variant: Model variant name.
        pretrained: If True, load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        SwinTransformerV2 model instance.
    """
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 1, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        SwinTransformerV2, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (8, 8),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
    ),
    'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth',
    ),
    'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),

    'swinv2_tiny_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth',
    ),
    'swinv2_tiny_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth',
    ),
    'swinv2_small_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth',
    ),
    'swinv2_small_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth',
    ),
    'swinv2_base_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth',
    ),
    'swinv2_base_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth',
    ),

    'swinv2_base_window12_192.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth',
        num_classes=21841, input_size=(3, 192, 192), pool_size=(6, 6)
    ),
    'swinv2_large_window12_192.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth',
        num_classes=21841, input_size=(3, 192, 192), pool_size=(6, 6)
    ),
})


@register_model
def swinv2_tiny_window16_256(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-T V2 @ 256x256, window 16x16."""
    model_args = dict(window_size=16, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_v2(
        'swinv2_tiny_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_tiny_window8_256(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-T V2 @ 256x256, window 8x8."""
    model_args = dict(window_size=8, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_v2(
        'swinv2_tiny_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_small_window16_256(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-S V2 @ 256x256, window 16x16."""
    model_args = dict(window_size=16, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_v2(
        'swinv2_small_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_small_window8_256(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-S V2 @ 256x256, window 8x8."""
    model_args = dict(window_size=8, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer_v2(
        'swinv2_small_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_base_window16_256(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-B V2 @ 256x256, window 16x16."""
    model_args = dict(window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer_v2(
        'swinv2_base_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_base_window8_256(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-B V2 @ 256x256, window 8x8."""
    model_args = dict(window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer_v2(
        'swinv2_base_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_base_window12_192(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-B V2 @ 192x192, window 12x12."""
    model_args = dict(window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer_v2(
        'swinv2_base_window12_192', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_base_window12to16_192to256(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-B V2 @ 192x192, trained at window 12x12, fine-tuned to 256x256 window 16x16."""
    model_args = dict(
        window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2(
        'swinv2_base_window12to16_192to256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_base_window12to24_192to384(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-B V2 @ 192x192, trained at window 12x12, fine-tuned to 384x384 window 24x24."""
    model_args = dict(
        window_size=24, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2(
        'swinv2_base_window12to24_192to384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_large_window12_192(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-L V2 @ 192x192, window 12x12."""
    model_args = dict(window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    return _create_swin_transformer_v2(
        'swinv2_large_window12_192', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_large_window12to16_192to256(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-L V2 @ 192x192, trained at window 12x12, fine-tuned to 256x256 window 16x16."""
    model_args = dict(
        window_size=16, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2(
        'swinv2_large_window12to16_192to256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_large_window12to24_192to384(pretrained: bool = False, **kwargs) -> SwinTransformerV2:
    """Swin-L V2 @ 192x192, trained at window 12x12, fine-tuned to 384x384 window 24x24."""
    model_args = dict(
        window_size=24, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2(
        'swinv2_large_window12to24_192to384', pretrained=pretrained, **dict(model_args, **kwargs))


register_model_deprecations(__name__, {
    'swinv2_base_window12_192_22k': 'swinv2_base_window12_192.ms_in22k',
    'swinv2_base_window12to16_192to256_22kft1k': 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
    'swinv2_base_window12to24_192to384_22kft1k': 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
    'swinv2_large_window12_192_22k': 'swinv2_large_window12_192.ms_in22k',
    'swinv2_large_window12to16_192to256_22kft1k': 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
    'swinv2_large_window12to24_192to384_22kft1k': 'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
})
