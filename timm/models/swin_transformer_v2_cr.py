""" Swin Transformer V2

A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/pdf/2111.09883

Code adapted from https://github.com/ChristophReich1996/Swin-Transformer-V2, original copyright/license info below

This implementation is experimental and subject to change in manners that will break weight compat:
* Size of the pos embed MLP are not spelled out in paper in terms of dim, fixed for all models? vary with num_heads?
  * currently dim is fixed, I feel it may make sense to scale with num_heads (dim per head)
* The specifics of the memory saving 'sequential attention' are not detailed, Christoph Reich has an impl at
  GitHub link above. It needs further investigation as throughput vs mem tradeoff doesn't appear beneficial.
* num_heads per stage is not detailed for Huge and Giant model variants
* 'Giant' is 3B params in paper but ~2.6B here despite matching paper dim + block counts
* experiments are ongoing wrt to 'main branch' norm layer use and weight init scheme

Noteworthy additions over official Swin v1:
* MLP relative position embedding is looking promising and adapts to different image/window sizes
* This impl has been designed to allow easy change of image size with matching window size changes
* Non-square image size and window size are supported

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer V2 reimplementation
# Copyright (c) 2021 Christoph Reich
# Licensed under The MIT License [see LICENSE for details]
# Written by Christoph Reich
# --------------------------------------------------------
import logging
import math
from typing import Tuple, Optional, List, Union, Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, Mlp, ClassifierHead, to_2tuple, _assert, ndgrid
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_function
from ._manipulate import named_apply
from ._registry import generate_default_cfgs, register_model

__all__ = ['SwinTransformerV2Cr']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C). """
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W). """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowMultiHeadAttention(nn.Module):
    r"""This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        meta_hidden_dim: int = 384,  # FIXME what's the optimal value?
        sequential_attn: bool = False,
    ) -> None:
        super(WindowMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (num_heads)."
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = to_2tuple(window_size)
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        # meta network for positional encodings
        self.meta_mlp = Mlp(
            2,  # x, y
            hidden_features=meta_hidden_dim,
            out_features=num_heads,
            act_layer=nn.ReLU,
            drop=(0.125, 0.)  # FIXME should there be stochasticity, appears to 'overfit' without?
        )
        # NOTE old checkpoints used inverse of logit_scale ('tau') following the paper, see conversion fn
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))
        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) -> None:
        """Method initializes the pair-wise relative positions to compute the positional biases."""
        device = self.logit_scale.device
        coordinates = torch.stack(ndgrid(
            torch.arange(self.window_size[0], device=device),
            torch.arange(self.window_size[1], device=device)
        ), dim=0).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(
            1.0 + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log, persistent=False)

    def set_window_size(self, window_size: Tuple[int, int]) -> None:
        """Update window size & interpolate position embeddings
        Args:
            window_size (int): New window size
        """
        window_size = to_2tuple(window_size)
        if window_size != self.window_size:
            self.window_size = window_size
            self._make_pair_wise_relative_positions()

    def _relative_positional_encodings(self) -> torch.Tensor:
        """Method computes the relative positional encodings

        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        window_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(
            self.num_heads, window_area, window_area
        )
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * windows, N, C)
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            Output tensor of the shape [B * windows, N, C]
        """
        Bw, L, C = x.shape

        qkv = self.qkv(x).view(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        # compute attention map with scaled cosine attention
        attn = (F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        attn = attn + self._relative_positional_encodings()

        if mask is not None:
            # Apply mask if utilized
            num_win: int = mask.shape[0]
            attn = attn.view(Bw // num_win, num_win, self.num_heads, L, L)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(Bw, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerV2CrBlock(nn.Module):
    r"""This class implements the Swin transformer block.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads to be utilized
        feat_size (Tuple[int, int]): Input resolution
        window_size (Tuple[int, int]): Window size to be utilized
        shift_size (int): Shifting size to be used
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        extra_norm (bool): Insert extra norm on 'main' branch if True
        sequential_attn (bool): If true sequential self-attention is performed
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            feat_size: Tuple[int, int],
            window_size: Tuple[int, int],
            shift_size: Tuple[int, int] = (0, 0),
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.0,
            init_values: Optional[float] = 0,
            proj_drop: float = 0.0,
            drop_attn: float = 0.0,
            drop_path: float = 0.0,
            extra_norm: bool = False,
            sequential_attn: bool = False,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super(SwinTransformerV2CrBlock, self).__init__()
        self.dim: int = dim
        self.feat_size: Tuple[int, int] = feat_size
        self.target_shift_size: Tuple[int, int] = to_2tuple(shift_size)
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.init_values: Optional[float] = init_values

        # attn branch
        self.attn = WindowMultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            drop_attn=drop_attn,
            drop_proj=proj_drop,
            sequential_attn=sequential_attn,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()

        # mlp branch
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
            out_features=dim,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()

        # Extra main branch norm layer mentioned for Huge/Giant models in V2 paper.
        # Also being used as final network norm and optional stage ending norm while still in a C-last format.
        self.norm3 = norm_layer(dim) if extra_norm else nn.Identity()

        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )
        self.init_weights()

    def _calc_window_shift(
            self,
            target_window_size: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_2tuple(target_window_size)
        target_shift_size = self.target_shift_size
        if any(target_shift_size):
            # if non-zero, recalculate shift from current window size in case window size has changed
            target_shift_size = (target_window_size[0] // 2, target_window_size[1] // 2)

        if self.always_partition:
            return target_window_size, target_shift_size

        window_size = [f if f <= w else w for f, w in zip(self.feat_size, target_window_size)]
        shift_size = [0 if f <= w else s for f, w, s in zip(self.feat_size, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def get_attn_mask(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Method generates the attention mask used in shift case."""
        # Make masks for shift case
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            if x is None:
                img_mask = torch.zeros((1, *self.feat_size, 1))  # 1 H W 1
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
            mask_windows = window_partition(img_mask, self.window_size)  # num_windows, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def init_weights(self):
        # extra, module specific weight init
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def set_input_size(self, feat_size: Tuple[int, int], window_size: Tuple[int, int]) -> None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            feat_size (Tuple[int, int]): New input resolution
            window_size (int): New window size
        """
        # Update input resolution
        self.feat_size: Tuple[int, int] = feat_size
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.set_window_size(self.window_size)
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def _shifted_window_attn(self, x):
        B, H, W, C = x.shape

        # cyclic shift
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)
        if do_shift:
            # FIXME PyTorch XLA needs cat impl, roll not lowered
            # x = torch.cat([x[:, sh:], x[:, :sh]], dim=1)
            # x = torch.cat([x[:, :, sw:], x[:, :, :sw]], dim=2)
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape

        # partition windows
        x_windows = window_partition(x, self.window_size)  # num_windows * B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        if getattr(self, 'dynamic_mask', False):
            attn_mask = self.get_attn_mask(x)
        else:
            attn_mask = self.attn_mask
        attn_windows = self.attn(x_windows, mask=attn_mask)  # num_windows * B, window_size * window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(attn_windows, self.window_size, (Hp, Wp))  # B H' W' C
        x = x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if do_shift:
            # FIXME PyTorch XLA needs cat impl, roll not lowered
            # x = torch.cat([x[:, -sh:], x[:, :-sh]], dim=1)
            # x = torch.cat([x[:, :, -sw:], x[:, :, :-sw]], dim=2)
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C, H, W]
        """
        # post-norm branches (op -> norm -> drop)
        x = x + self.drop_path1(self.norm1(self._shifted_window_attn(x)))

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = self.norm3(x)  # main-branch norm enabled for some blocks / stages (every 6 for Huge/Giant)
        x = x.reshape(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    """ This class implements the patch merging as a strided convolution with a normalization before.
    Args:
        dim (int): Number of input channels
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized.
    """

    def __init__(self, dim: int, norm_layer: Type[nn.Module] = nn.LayerNorm) -> None:
        super(PatchMerging, self).__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(in_features=4 * dim, out_features=2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            strict_img_size=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def set_input_size(self, img_size: Tuple[int, int]):
        img_size = to_2tuple(img_size)
        if img_size != self.img_size:
            self.img_size = img_size
            self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.strict_img_size:
            _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class SwinTransformerV2CrStage(nn.Module):
    r"""This class implements a stage of the Swin transformer including multiple layers.

    Args:
        embed_dim (int): Number of input channels
        depth (int): Depth of the stage (number of layers)
        downscale (bool): If true input is downsampled (see Fig. 3 or V1 paper)
        feat_size (Tuple[int, int]): input feature map size (H, W)
        num_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        downscale: bool,
        num_heads: int,
        feat_size: Tuple[int, int],
        window_size: Tuple[int, int],
        always_partition: bool = False,
        dynamic_mask: bool = False,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.0,
        proj_drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: Union[List[float], float] = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
    ):
        super(SwinTransformerV2CrStage, self).__init__()
        self.downscale: bool = downscale
        self.grad_checkpointing: bool = False
        self.feat_size: Tuple[int, int] = (feat_size[0] // 2, feat_size[1] // 2) if downscale else feat_size

        if downscale:
            self.downsample = PatchMerging(embed_dim, norm_layer=norm_layer)
            embed_dim = embed_dim * 2
        else:
            self.downsample = nn.Identity()

        def _extra_norm(index):
            i = index + 1
            if extra_norm_period and i % extra_norm_period == 0:
                return True
            return i == depth if extra_norm_stage else False

        self.blocks = nn.Sequential(*[
            SwinTransformerV2CrBlock(
                dim=embed_dim,
                num_heads=num_heads,
                feat_size=self.feat_size,
                window_size=window_size,
                always_partition=always_partition,
                dynamic_mask=dynamic_mask,
                shift_size=tuple([0 if ((index % 2) == 0) else w // 2 for w in window_size]),
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop,
                drop_attn=drop_attn,
                drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                extra_norm=_extra_norm(index),
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
            )
            for index in range(depth)]
        )

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: int,
            always_partition: Optional[bool] = None,
    ):
        """ Updates the resolution to utilize and the window size and so the pair-wise relative positions.

        Args:
            window_size (int): New window size
            feat_size (Tuple[int, int]): New input resolution
        """
        self.feat_size = (feat_size[0] // 2, feat_size[1] // 2) if self.downscale else feat_size
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.feat_size,
                window_size=window_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W] or [B, L, C]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        x = bchw_to_bhwc(x)
        x = self.downsample(x)
        for block in self.blocks:
            # Perform checkpointing if utilized
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        x = bhwc_to_bchw(x)
        return x


class SwinTransformerV2Cr(nn.Module):
    r""" Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`  -
          https://arxiv.org/pdf/2111.09883

    Args:
        img_size: Input resolution.
        window_size: Window size. If None, grid_size // window_div
        window_ratio: Window size to patch grid ratio.
        patch_size: Patch size.
        in_chans: Number of input channels.
        depths: Depth of the stage (number of layers).
        num_heads: Number of attention heads to be utilized.
        embed_dim: Patch embedding dimension.
        num_classes: Number of output classes.
        mlp_ratio:  Ratio of the hidden dimension in the FFN to the input channels.
        drop_rate: Dropout rate.
        proj_drop_rate: Projection dropout rate.
        attn_drop_rate: Dropout rate of attention map.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Type of normalization layer to be utilized.
        extra_norm_period: Insert extra norm layer on main branch every N (period) blocks in stage
        extra_norm_stage: End each stage with an extra norm layer in main branch
        sequential_attn: If true sequential self-attention is performed.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        window_size: Optional[int] = None,
        window_ratio: int = 8,
        always_partition: bool = False,
        strict_img_size: bool = True,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
        global_pool: str = 'avg',
        weight_init='skip',
        **kwargs: Any
    ) -> None:
        super(SwinTransformerV2Cr, self).__init__()
        img_size = to_2tuple(img_size)
        self.num_classes: int = num_classes
        self.patch_size: int = patch_size
        self.img_size: Tuple[int, int] = img_size
        self.num_features = self.head_hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        self.feature_info = []

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            strict_img_size=strict_img_size,
        )
        grid_size = self.patch_embed.grid_size
        if window_size is None:
            self.window_size = tuple([s // window_ratio for s in grid_size])
        else:
            self.window_size = to_2tuple(window_size)

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        in_dim = embed_dim
        in_scale = 1
        for stage_idx, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stages += [SwinTransformerV2CrStage(
                embed_dim=in_dim,
                depth=depth,
                downscale=stage_idx != 0,
                feat_size=(grid_size[0] // in_scale, grid_size[1] // in_scale),
                num_heads=num_heads,
                window_size=self.window_size,
                always_partition=always_partition,
                dynamic_mask=not strict_img_size,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                drop_attn=attn_drop_rate,
                drop_path=dpr[stage_idx],
                extra_norm_period=extra_norm_period,
                extra_norm_stage=extra_norm_stage or (stage_idx + 1) == len(depths),  # last stage ends w/ norm
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
            )]
            if stage_idx != 0:
                in_dim *= 2
                in_scale *= 2
            self.feature_info += [dict(num_chs=in_dim, reduction=4 * in_scale, module=f'stages.{stage_idx}')]
        self.stages = nn.Sequential(*stages)

        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
        )

        # current weight init skips custom init and uses pytorch layer defaults, seems to work well
        # FIXME more experiments needed
        if weight_init != 'skip':
            named_apply(init_weights, self)

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            window_size: Optional[Tuple[int, int]] = None,
            window_ratio: int = 8,
            always_partition: Optional[bool] = None,
    ) -> None:
        """Updates the image resolution, window size and so the pair-wise relative positions.

        Args:
            img_size (Optional[Tuple[int, int]]): New input resolution, if None current resolution is used
            window_size (Optional[int]): New window size, if None based on new_img_size // window_div
            window_ratio (int): divisor for calculating window size from patch grid size
            always_partition: always partition / shift windows even if feat size is < window
        """
        if img_size is not None:
            self.patch_embed.set_input_size(img_size=img_size)
            grid_size = self.patch_embed.grid_size

        if window_size is None and window_ratio is not None:
            window_size = tuple([s // window_ratio for s in grid_size])

        for index, stage in enumerate(self.stages):
            stage_scale = 2 ** max(index - 1, 0)
            stage.set_input_size(
                feat_size=(grid_size[0] // stage_scale, grid_size[1] // stage_scale),
                window_size=window_size,
                always_partition=always_partition,
            )

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore()
    def get_classifier(self) -> nn.Module:
        """Method returns the classification head of the model.
        Returns:
            head (nn.Module): Current classification head
        """
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Method results the classification head

        Args:
            num_classes (int): Number of classes to be predicted
            global_pool (str): Unused
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
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.patch_embed(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]
        for i, stage in enumerate(stages):
            x = stage(x)
            if i in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def init_weights(module: nn.Module, name: str = ''):
    # FIXME WIP determining if there's a better weight init
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        elif 'head' in name:
            nn.init.zeros_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    if 'head.fc.weight' in state_dict:
        return state_dict
    out_dict = {}
    for k, v in state_dict.items():
        if 'tau' in k:
            # convert old tau based checkpoints -> logit_scale (inverse)
            v = torch.log(1 / v)
            k = k.replace('tau', 'logit_scale')
        k = k.replace('head.', 'head.fc.')
        out_dict[k] = v
    return out_dict


def _create_swin_transformer_v2_cr(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 1, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        SwinTransformerV2Cr, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head.fc',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'swinv2_cr_tiny_384.untrained': _cfg(
        url="", input_size=(3, 384, 384), crop_pct=1.0, pool_size=(12, 12)),
    'swinv2_cr_tiny_224.untrained': _cfg(
        url="", input_size=(3, 224, 224), crop_pct=0.9),
    'swinv2_cr_tiny_ns_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-swinv2/swin_v2_cr_tiny_ns_224-ba8166c6.pth",
        input_size=(3, 224, 224), crop_pct=0.9),
    'swinv2_cr_small_384.untrained': _cfg(
        url="", input_size=(3, 384, 384), crop_pct=1.0, pool_size=(12, 12)),
    'swinv2_cr_small_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-swinv2/swin_v2_cr_small_224-0813c165.pth",
        input_size=(3, 224, 224), crop_pct=0.9),
    'swinv2_cr_small_ns_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-swinv2/swin_v2_cr_small_ns_224_iv-2ce90f8e.pth",
        input_size=(3, 224, 224), crop_pct=0.9),
    'swinv2_cr_small_ns_256.untrained': _cfg(
        url="", input_size=(3, 256, 256), crop_pct=1.0, pool_size=(8, 8)),
    'swinv2_cr_base_384.untrained': _cfg(
        url="", input_size=(3, 384, 384), crop_pct=1.0, pool_size=(12, 12)),
    'swinv2_cr_base_224.untrained': _cfg(
        url="", input_size=(3, 224, 224), crop_pct=0.9),
    'swinv2_cr_base_ns_224.untrained': _cfg(
        url="", input_size=(3, 224, 224), crop_pct=0.9),
    'swinv2_cr_large_384.untrained': _cfg(
        url="", input_size=(3, 384, 384), crop_pct=1.0, pool_size=(12, 12)),
    'swinv2_cr_large_224.untrained': _cfg(
        url="", input_size=(3, 224, 224), crop_pct=0.9),
    'swinv2_cr_huge_384.untrained': _cfg(
        url="", input_size=(3, 384, 384), crop_pct=1.0, pool_size=(12, 12)),
    'swinv2_cr_huge_224.untrained': _cfg(
        url="", input_size=(3, 224, 224), crop_pct=0.9),
    'swinv2_cr_giant_384.untrained': _cfg(
        url="", input_size=(3, 384, 384), crop_pct=1.0, pool_size=(12, 12)),
    'swinv2_cr_giant_224.untrained': _cfg(
        url="", input_size=(3, 224, 224), crop_pct=0.9),
})


@register_model
def swinv2_cr_tiny_384(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-T V2 CR @ 384x384, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_tiny_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_tiny_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-T V2 CR @ 224x224, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_tiny_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_tiny_ns_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-T V2 CR @ 224x224, trained ImageNet-1k w/ extra stage norms.
    ** Experimental, may make default if results are improved. **
    """
    model_args = dict(
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        extra_norm_stage=True,
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_tiny_ns_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_small_384(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-S V2 CR @ 384x384, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_small_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_small_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-S V2 CR @ 224x224, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_small_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_small_ns_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-S V2 CR @ 224x224, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        extra_norm_stage=True,
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_small_ns_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_small_ns_256(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-S V2 CR @ 256x256, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        extra_norm_stage=True,
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_small_ns_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_base_384(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-B V2 CR @ 384x384, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_base_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_base_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-B V2 CR @ 224x224, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_base_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_base_ns_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-B V2 CR @ 224x224, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        extra_norm_stage=True,
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_base_ns_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_large_384(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-L V2 CR @ 384x384, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_large_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_large_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-L V2 CR @ 224x224, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_large_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_huge_384(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-H V2 CR @ 384x384, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=352,
        depths=(2, 2, 18, 2),
        num_heads=(11, 22, 44, 88),  # head count not certain for Huge, 384 & 224 trying diff values
        extra_norm_period=6,
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_huge_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_huge_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-H V2 CR @ 224x224, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=352,
        depths=(2, 2, 18, 2),
        num_heads=(8, 16, 32, 64),  # head count not certain for Huge, 384 & 224 trying diff values
        extra_norm_period=6,
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_huge_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_giant_384(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-G V2 CR @ 384x384, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=512,
        depths=(2, 2, 42, 2),
        num_heads=(16, 32, 64, 128),
        extra_norm_period=6,
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_giant_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swinv2_cr_giant_224(pretrained=False, **kwargs) -> SwinTransformerV2Cr:
    """Swin-G V2 CR @ 224x224, trained ImageNet-1k"""
    model_args = dict(
        embed_dim=512,
        depths=(2, 2, 42, 2),
        num_heads=(16, 32, 64, 128),
        extra_norm_period=6,
    )
    return _create_swin_transformer_v2_cr('swinv2_cr_giant_224', pretrained=pretrained, **dict(model_args, **kwargs))
