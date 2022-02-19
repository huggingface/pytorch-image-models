""" Swin Transformer V2
A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/pdf/2111.09883

Code adapted from https://github.com/ChristophReich1996/Swin-Transformer-V2, original copyright/license info below

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer V2 reimplementation
# Copyright (c) 2021 Christoph Reich
# Licensed under The MIT License [see LICENSE for details]
# Written by Christoph Reich
# --------------------------------------------------------
import logging
from copy import deepcopy
from typing import Tuple, Optional, List, Union, Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from .helpers import build_model_with_cfg, overlay_external_default_cfg
# from .vision_transformer import checkpoint_filter_fn
# from .registry import register_model
# from .layers import DropPath, Mlp

from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.vision_transformer import checkpoint_filter_fn
from timm.models.registry import register_model
from timm.models.layers import DropPath, Mlp

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'swin_v2_cr_tiny_patch4_window12_384': _cfg(
        url="",
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_v2_cr_tiny_patch4_window7_224': _cfg(
        url="",
        input_size=(3, 224, 224), crop_pct=1.0),

    'swin_v2_cr_small_patch4_window12_384': _cfg(
        url="",
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_v2_cr_small_patch4_window7_224': _cfg(
        url="",
        input_size=(3, 224, 224), crop_pct=1.0),

    'swin_v2_cr_base_patch4_window12_384': _cfg(
        url="",
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_v2_cr_base_patch4_window7_224': _cfg(
        url="",
        input_size=(3, 224, 224), crop_pct=1.0),

    'swin_v2_cr_large_patch4_window12_384': _cfg(
        url="",
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_v2_cr_large_patch4_window7_224': _cfg(
        url="",
        input_size=(3, 224, 224), crop_pct=1.0),

    'swin_v2_cr_huge_patch4_window12_384': _cfg(
        url="",
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_v2_cr_huge_patch4_window7_224': _cfg(
        url="",
        input_size=(3, 224, 224), crop_pct=1.0),

    'swin_v2_cr_giant_patch4_window12_384': _cfg(
        url="",
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_v2_cr_giant_patch4_window7_224': _cfg(
        url="",
        input_size=(3, 224, 224), crop_pct=1.0),
}


def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """ Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C).

    Args:
        input (torch.Tensor): Input tensor of the shape (B, C, H, W)

    Returns:
        output (torch.Tensor): Permuted tensor of the shape (B, H, W, C)
    """
    output: torch.Tensor = input.permute(0, 2, 3, 1)
    return output


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """ Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W).

    Args:
        input (torch.Tensor): Input tensor of the shape (B, H, W, C)

    Returns:
        output (torch.Tensor): Permuted tensor of the shape (B, C, H, W)
    """
    output: torch.Tensor = input.permute(0, 3, 1, 2)
    return output


def unfold(input: torch.Tensor,
           window_size: int) -> torch.Tensor:
    """ Unfolds (non-overlapping) a given feature map by the given window size (stride = window size).

    Args:
        input (torch.Tensor): Input feature map of the shape (B, C, H, W)
        window_size (int): Window size to be applied

    Returns:
        output (torch.Tensor): Unfolded tensor of the shape [B * windows, C, window size, window size]
    """
    # Get original shape
    _, channels, height, width = input.shape  # type: int, int, int, int
    # Unfold input
    output: torch.Tensor = input.unfold(dimension=3, size=window_size, step=window_size) \
        .unfold(dimension=2, size=window_size, step=window_size)
    # Reshape to (B * windows, C, window size, window size)
    output: torch.Tensor = output.permute(0, 2, 3, 1, 5, 4).reshape(-1, channels, window_size, window_size)
    return output


def fold(input: torch.Tensor,
         window_size: int,
         height: int,
         width: int) -> torch.Tensor:
    """ Folds a tensor of windows again to a 4D feature map.

    Args:
        input (torch.Tensor): Input feature map of the shape (B, C, H, W)
        window_size (int): Window size of the unfold operation
        height (int): Height of the feature map
        width (int): Width of the feature map

    Returns:
        output (torch.Tensor): Folded output tensor of the shape (B, C, H, W)
    """
    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    batch_size: int = int(input.shape[0] // (height * width // window_size // window_size))
    # Reshape input to (B, C, H, W)
    output: torch.Tensor = input.view(batch_size, height // window_size, width // window_size, channels,
                                      window_size, window_size)
    output: torch.Tensor = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
    return output


class WindowMultiHeadAttention(nn.Module):
    r""" This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        in_features (int): Number of input features
        window_size (int): Window size
        number_of_heads (int): Number of attention heads
        dropout_attention (float): Dropout rate of attention map
        dropout_projection (float): Dropout rate after projection
        meta_network_hidden_features (int): Number of hidden features in the two layer MLP meta network
        sequential_self_attention (bool): If true sequential self-attention is performed
    """

    def __init__(self,
                 in_features: int,
                 window_size: int,
                 number_of_heads: int,
                 dropout_attention: float = 0.,
                 dropout_projection: float = 0.,
                 meta_network_hidden_features: int = 256,
                 sequential_self_attention: bool = False) -> None:
        # Call super constructor
        super(WindowMultiHeadAttention, self).__init__()
        # Check parameter
        assert (in_features % number_of_heads) == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (number_of_heads)."
        # Save parameters
        self.in_features: int = in_features
        self.window_size: int = window_size
        self.number_of_heads: int = number_of_heads
        self.sequential_self_attention: bool = sequential_self_attention
        # Init query, key and value mapping as a single layer
        self.mapping_qkv: nn.Module = nn.Linear(in_features=in_features, out_features=in_features * 3, bias=True)
        # Init attention dropout
        self.attention_dropout: nn.Module = nn.Dropout(dropout_attention)
        # Init projection mapping
        self.projection: nn.Module = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        # Init projection dropout
        self.projection_dropout: nn.Module = nn.Dropout(dropout_projection)
        # Init meta network for positional encodings
        self.meta_network: nn.Module = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_network_hidden_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_network_hidden_features, out_features=number_of_heads, bias=True))
        # Init tau
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(1, number_of_heads, 1, 1)))
        # Init pair-wise relative positions (log-spaced)
        self.__make_pair_wise_relative_positions()

    def __make_pair_wise_relative_positions(self) -> None:
        """ Method initializes the pair-wise relative positions to compute the positional biases."""
        indexes: torch.Tensor = torch.arange(self.window_size, device=self.tau.device)
        coordinates: torch.Tensor = torch.stack(torch.meshgrid([indexes, indexes]), dim=0)
        coordinates: torch.Tensor = torch.flatten(coordinates, start_dim=1)
        relative_coordinates: torch.Tensor = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates: torch.Tensor = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log: torch.Tensor = torch.sign(relative_coordinates) \
                                                 * torch.log(1. + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log)

    def update_resolution(self,
                          new_window_size: int,
                          **kwargs: Any) -> None:
        """ Method updates the window size and so the pair-wise relative positions

        Args:
            new_window_size (int): New window size
            kwargs (Any): Unused
        """
        # Set new window size
        self.window_size: int = new_window_size
        # Make new pair-wise relative positions
        self.__make_pair_wise_relative_positions()

    def __get_relative_positional_encodings(self) -> torch.Tensor:
        """ Method computes the relative positional encodings

        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        relative_position_bias: torch.Tensor = self.meta_network(self.relative_coordinates_log)
        relative_position_bias: torch.Tensor = relative_position_bias.permute(1, 0)
        relative_position_bias: torch.Tensor = relative_position_bias.reshape(self.number_of_heads,
                                                                              self.window_size * self.window_size,
                                                                              self.window_size * self.window_size)
        relative_position_bias: torch.Tensor = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def __self_attention(self,
                         query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         batch_size_windows: int,
                         tokens: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ This function performs standard (non-sequential) scaled cosine self-attention.

        Args:
            query (torch.Tensor): Query tensor of the shape [B * windows, heads, tokens, C // heads]
            key (torch.Tensor): Key tensor of the shape [B * windows, heads, tokens, C // heads]
            value (torch.Tensor): Value tensor of the shape (B * windows, heads, tokens, C // heads)
            batch_size_windows (int): Size of the first dimension of the input tensor batch size * windows
            tokens (int): Number of tokens in the input
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            output (torch.Tensor): Output feature map of the shape [B * windows, tokens, C]
        """
        # Compute attention map with scaled cosine attention
        attention_map: torch.Tensor = torch.einsum("bhqd, bhkd -> bhqk", query, key) \
                                      / torch.maximum(torch.norm(query, dim=-1, keepdim=True)
                                                      * torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1),
                                                      torch.tensor(1e-06, device=query.device, dtype=query.dtype))
        attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)
        # Apply relative positional encodings
        attention_map: torch.Tensor = attention_map + self.__get_relative_positional_encodings()
        # Apply mask if utilized
        if mask is not None:
            number_of_windows: int = mask.shape[0]
            attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows, number_of_windows,
                                                             self.number_of_heads, tokens, tokens)
            attention_map: torch.Tensor = attention_map + mask.unsqueeze(1).unsqueeze(0)
            attention_map: torch.Tensor = attention_map.view(-1, self.number_of_heads, tokens, tokens)
        attention_map: torch.Tensor = attention_map.softmax(dim=-1)
        # Perform attention dropout
        attention_map: torch.Tensor = self.attention_dropout(attention_map)
        # Apply attention map and reshape
        output: torch.Tensor = torch.einsum("bhal, bhlv -> bhav", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def __sequential_self_attention(self,
                                    query: torch.Tensor,
                                    key: torch.Tensor,
                                    value: torch.Tensor,
                                    batch_size_windows: int,
                                    tokens: int,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ This function performs sequential scaled cosine self-attention.

        Args:
            query (torch.Tensor): Query tensor of the shape [B * windows, heads, tokens, C // heads]
            key (torch.Tensor): Key tensor of the shape [B * windows, heads, tokens, C // heads]
            value (torch.Tensor): Value tensor of the shape (B * windows, heads, tokens, C // heads)
            batch_size_windows (int): Size of the first dimension of the input tensor batch size * windows
            tokens (int): Number of tokens in the input
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            output (torch.Tensor): Output feature map of the shape [B * windows, tokens, C]
        """
        # Init output tensor
        output: torch.Tensor = torch.ones_like(query)
        # Compute relative positional encodings fist
        relative_position_bias: torch.Tensor = self.__get_relative_positional_encodings()
        # Iterate over query and key tokens
        for token_index_query in range(tokens):
            # Compute attention map with scaled cosine attention
            attention_map: torch.Tensor = \
                torch.einsum("bhd, bhkd -> bhk", query[:, :, token_index_query], key) \
                / torch.maximum(torch.norm(query[:, :, token_index_query], dim=-1, keepdim=True)
                                * torch.norm(key, dim=-1, keepdim=False),
                                torch.tensor(1e-06, device=query.device, dtype=query.dtype))
            attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)[..., 0]
            # Apply positional encodings
            attention_map: torch.Tensor = attention_map + relative_position_bias[..., token_index_query, :]
            # Apply mask if utilized
            if mask is not None:
                number_of_windows: int = mask.shape[0]
                attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows,
                                                                 number_of_windows, self.number_of_heads, 1,
                                                                 tokens)
                attention_map: torch.Tensor = attention_map \
                                              + mask.unsqueeze(1).unsqueeze(0)[..., token_index_query, :].unsqueeze(3)
                attention_map: torch.Tensor = attention_map.view(-1, self.number_of_heads, tokens)
            attention_map: torch.Tensor = attention_map.softmax(dim=-1)
            # Perform attention dropout
            attention_map: torch.Tensor = self.attention_dropout(attention_map)
            # Apply attention map and reshape
            output[:, :, token_index_query] = torch.einsum("bhl, bhlv -> bhv", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape (B * windows, C, H, W)
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            output (torch.Tensor): Output tensor of the shape [B * windows, C, H, W]
        """
        # Save original shape
        batch_size_windows, channels, height, width = input.shape  # type: int, int, int, int
        tokens: int = height * width
        # Reshape input to (B * windows, tokens (height * width), C)
        input: torch.Tensor = input.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
        # Perform query, key, and value mapping
        query_key_value: torch.Tensor = self.mapping_qkv(input)
        query_key_value: torch.Tensor = query_key_value.view(batch_size_windows, tokens, 3, self.number_of_heads,
                                                             channels // self.number_of_heads).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]
        # Perform attention
        if self.sequential_self_attention:
            output: torch.Tensor = self.__sequential_self_attention(query=query, key=key, value=value,
                                                                    batch_size_windows=batch_size_windows,
                                                                    tokens=tokens,
                                                                    mask=mask)
        else:
            output: torch.Tensor = self.__self_attention(query=query, key=key, value=value,
                                                         batch_size_windows=batch_size_windows, tokens=tokens,
                                                         mask=mask)
        # Perform linear mapping and dropout
        output: torch.Tensor = self.projection_dropout(self.projection(output))
        # Reshape output to original shape [B * windows, C, H, W]
        output: torch.Tensor = output.permute(0, 2, 1).view(batch_size_windows, channels, height, width)
        return output


class SwinTransformerBlock(nn.Module):
    r""" This class implements the Swin transformer block.

    Args:
        in_channels (int): Number of input channels
        input_resolution (Tuple[int, int]): Input resolution
        number_of_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        shift_size (int): Shifting size to be used
        ff_feature_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        dropout (float): Dropout in input mapping
        dropout_attention (float): Dropout rate of attention map
        dropout_path (float): Dropout in main path
        sequential_self_attention (bool): If true sequential self-attention is performed
    """

    def __init__(self,
                 in_channels: int,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False) -> None:
        # Call super constructor
        super(SwinTransformerBlock, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.input_resolution: Tuple[int, int] = input_resolution
        # Catch case if resolution is smaller than the window size
        if min(self.input_resolution) <= window_size:
            self.window_size: int = min(self.input_resolution)
            self.shift_size: int = 0
            self.make_windows: bool = False
        else:
            self.window_size: int = window_size
            self.shift_size: int = shift_size
            self.make_windows: bool = True
        # Init normalization layers
        self.normalization_1: nn.Module = nn.LayerNorm(normalized_shape=in_channels)
        self.normalization_2: nn.Module = nn.LayerNorm(normalized_shape=in_channels)
        # Init window attention module
        self.window_attention: WindowMultiHeadAttention = WindowMultiHeadAttention(
            in_features=in_channels,
            window_size=self.window_size,
            number_of_heads=number_of_heads,
            dropout_attention=dropout_attention,
            dropout_projection=dropout,
            sequential_self_attention=sequential_self_attention)
        # Init dropout layer
        self.dropout: nn.Module = DropPath(drop_prob=dropout_path) if dropout_path > 0. else nn.Identity()
        # Init feed-forward network
        self.feed_forward_network: nn.Module = Mlp(in_features=in_channels,
                                                   hidden_features=int(in_channels * ff_feature_ratio),
                                                   drop=dropout,
                                                   out_features=in_channels)
        # Make attention mask
        self.__make_attention_mask()

    def __make_attention_mask(self) -> None:
        """ Method generates the attention mask used in shift case. """
        # Make masks for shift case
        if self.shift_size > 0:
            height, width = self.input_resolution  # type: int, int
            mask: torch.Tensor = torch.zeros(height, width, device=self.window_attention.tau.device)
            height_slices: Tuple = (slice(0, -self.window_size),
                                    slice(-self.window_size, -self.shift_size),
                                    slice(-self.shift_size, None))
            width_slices: Tuple = (slice(0, -self.window_size),
                                   slice(-self.window_size, -self.shift_size),
                                   slice(-self.shift_size, None))
            counter: int = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    mask[height_slice, width_slice] = counter
                    counter += 1
            mask_windows: torch.Tensor = unfold(mask[None, None], self.window_size)
            mask_windows: torch.Tensor = mask_windows.reshape(-1, self.window_size * self.window_size)
            attention_mask: Optional[torch.Tensor] = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask == 0, float(0.0))
        else:
            attention_mask: Optional[torch.Tensor] = None
        # Save mask
        self.register_buffer("attention_mask", attention_mask)

    def update_resolution(self,
                          new_window_size: int,
                          new_input_resolution: Tuple[int, int]) -> None:
        """ Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_input_resolution (Tuple[int, int]): New input resolution
        """
        # Update input resolution
        self.input_resolution: Tuple[int, int] = new_input_resolution
        # Catch case if resolution is smaller than the window size
        if min(self.input_resolution) <= new_window_size:
            self.window_size: int = min(self.input_resolution)
            self.shift_size: int = 0
            self.make_windows: bool = False
        else:
            self.window_size: int = new_window_size
            self.shift_size: int = self.shift_size
            self.make_windows: bool = True
        # Update attention mask
        self.__make_attention_mask()
        # Update attention module
        self.window_attention.update_resolution(new_window_size=new_window_size)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C, H, W]
        """
        # Save shape
        batch_size, channels, height, width = input.shape  # type: int, int, int, int
        # Shift input if utilized
        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=input, shifts=(-self.shift_size, -self.shift_size),
                                                    dims=(-1, -2))
        else:
            output_shift: torch.Tensor = input
        # Make patches
        output_patches: torch.Tensor = unfold(input=output_shift, window_size=self.window_size) \
            if self.make_windows else output_shift
        # Perform window attention
        output_attention: torch.Tensor = self.window_attention(output_patches, mask=self.attention_mask)
        # Merge patches
        output_merge: torch.Tensor = fold(input=output_attention, window_size=self.window_size, height=height,
                                          width=width) if self.make_windows else output_attention
        # Reverse shift if utilized
        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=output_merge, shifts=(self.shift_size, self.shift_size),
                                                    dims=(-1, -2))
        else:
            output_shift: torch.Tensor = output_merge
        # Perform normalization
        output_normalize: torch.Tensor = self.normalization_1(output_shift.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Skip connection
        output_skip: torch.Tensor = self.dropout(output_normalize) + input
        # Feed forward network, normalization and skip connection
        output_feed_forward: torch.Tensor = self.feed_forward_network(
            output_skip.view(batch_size, channels, -1).permute(0, 2, 1)).permute(0, 2, 1)
        output_feed_forward: torch.Tensor = output_feed_forward.view(batch_size, channels, height, width)
        output_normalize: torch.Tensor = bhwc_to_bchw(self.normalization_2(bchw_to_bhwc(output_feed_forward)))
        output: torch.Tensor = output_skip + self.dropout(output_normalize)
        return output


class DeformableSwinTransformerBlock(SwinTransformerBlock):
    r""" This class implements a deformable version of the Swin Transformer block.
        Inspired by: https://arxiv.org/pdf/2201.00520

    Args:
        in_channels (int): Number of input channels
        input_resolution (Tuple[int, int]): Input resolution
        number_of_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        shift_size (int): Shifting size to be used
        ff_feature_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        dropout (float): Dropout in input mapping
        dropout_attention (float): Dropout rate of attention map
        dropout_path (float): Dropout in main path
        sequential_self_attention (bool): If true sequential self-attention is performed
        offset_downscale_factor (int): Downscale factor of offset network
    """

    def __init__(self,
                 in_channels: int,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False,
                 offset_downscale_factor: int = 2) -> None:
        # Call super constructor
        super(DeformableSwinTransformerBlock, self).__init__(
            in_channels=in_channels,
            input_resolution=input_resolution,
            number_of_heads=number_of_heads,
            window_size=window_size,
            shift_size=shift_size,
            ff_feature_ratio=ff_feature_ratio,
            dropout=dropout,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path,
            sequential_self_attention=sequential_self_attention
        )
        # Save parameter
        self.offset_downscale_factor: int = offset_downscale_factor
        self.number_of_heads: int = number_of_heads
        # Make default offsets
        self.__make_default_offsets()
        # Init offset network
        self.offset_network: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=offset_downscale_factor,
                      padding=3, groups=in_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=2 * self.number_of_heads, kernel_size=1, stride=1,
                      padding=0, bias=True)
        )

    def __make_default_offsets(self) -> None:
        """ Method generates the default sampling grid (inspired by kornia) """
        # Init x and y coordinates
        x: torch.Tensor = torch.linspace(0, self.input_resolution[1] - 1, self.input_resolution[1],
                                         device=self.window_attention.tau.device)
        y: torch.Tensor = torch.linspace(0, self.input_resolution[0] - 1, self.input_resolution[0],
                                         device=self.window_attention.tau.device)
        # Normalize coordinates to a range of [-1, 1]
        x: torch.Tensor = (x / (self.input_resolution[1] - 1) - 0.5) * 2
        y: torch.Tensor = (y / (self.input_resolution[0] - 1) - 0.5) * 2
        # Make grid [2, height, width]
        grid: torch.Tensor = torch.stack(torch.meshgrid([x, y])).transpose(1, 2)
        # Reshape grid to [1, height, width, 2]
        grid: torch.Tensor = grid.unsqueeze(dim=0).permute(0, 2, 3, 1)
        # Register in module
        self.register_buffer("default_grid", grid)

    def update_resolution(self,
                          new_window_size: int,
                          new_input_resolution: Tuple[int, int]) -> None:
        """ Method updates the window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_input_resolution (Tuple[int, int]): New input resolution
        """
        # Update resolution and window size
        super(DeformableSwinTransformerBlock, self).update_resolution(new_window_size=new_window_size,
                                                                      new_input_resolution=new_input_resolution)
        # Update default sampling grid
        self.__make_default_offsets()

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """ Forward pass
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C, H, W]
        """
        # Get input shape
        batch_size, channels, height, width = input.shape
        # Compute offsets of the shape [batch size, 2, height / r, width / r]
        offsets: torch.Tensor = self.offset_network(input)
        # Upscale offsets to the shape [batch size, 2 * number of heads, height, width]
        offsets: torch.Tensor = F.interpolate(input=offsets,
                                              size=(height, width), mode="bilinear", align_corners=True)
        # Reshape offsets to [batch size, number of heads, height, width, 2]
        offsets: torch.Tensor = offsets.reshape(batch_size, -1, 2, height, width).permute(0, 1, 3, 4, 2)
        # Flatten batch size and number of heads and apply tanh
        offsets: torch.Tensor = offsets.view(-1, height, width, 2).tanh()
        # Cast offset grid to input data type
        if input.dtype != self.default_grid.dtype:
            self.default_grid = self.default_grid.type(input.dtype)
        # Construct offset grid
        offset_grid: torch.Tensor = self.default_grid.repeat_interleave(repeats=offsets.shape[0], dim=0) + offsets
        # Reshape input to [batch size * number of heads, channels / number of heads, height, width]
        input: torch.Tensor = input.view(batch_size, self.number_of_heads, channels // self.number_of_heads, height,
                                         width).flatten(start_dim=0, end_dim=1)
        # Apply sampling grid
        input_resampled: torch.Tensor = F.grid_sample(input=input, grid=offset_grid.clip(min=-1, max=1),
                                                      mode="bilinear", align_corners=True, padding_mode="reflection")
        # Reshape resampled tensor again to [batch size, channels, height, width]
        input_resampled: torch.Tensor = input_resampled.view(batch_size, channels, height, width)
        output: torch.Tensor = super(DeformableSwinTransformerBlock, self).forward(input=input_resampled)
        return output


class PatchMerging(nn.Module):
    """ This class implements the patch merging as a strided convolution with a normalization before.

    Args:
        in_channels (int): Number of input channels
    """

    def __init__(self,
                 in_channels: int) -> None:
        # Call super constructor
        super(PatchMerging, self).__init__()
        # Init normalization
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=4 * in_channels)
        # Init linear mapping
        self.linear_mapping: nn.Module = nn.Linear(in_features=4 * in_channels, out_features=2 * in_channels,
                                                   bias=False)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        # Get original shape
        batch_size, channels, height, width = input.shape  # type: int, int, int, int
        # Reshape input to [batch size, in channels, height, width]
        input: torch.Tensor = bchw_to_bhwc(input)
        # Unfold input
        input: torch.Tensor = input.unfold(dimension=1, size=2, step=2).unfold(dimension=2, size=2, step=2)
        input: torch.Tensor = input.reshape(batch_size, input.shape[1], input.shape[2], -1)
        # Normalize input
        input: torch.Tensor = self.normalization(input)
        # Perform linear mapping
        output: torch.Tensor = bhwc_to_bchw(self.linear_mapping(input))
        return output


class PatchEmbedding(nn.Module):
    """ Module embeds a given image into patch embeddings.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        patch_size (int): Patch size to be utilized
        image_size (int): Image size to be used
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 96,
                 patch_size: int = 4) -> None:
        # Call super constructor
        super(PatchEmbedding, self).__init__()
        # Save parameters
        self.out_channels: int = out_channels
        # Init linear embedding as a convolution
        self.linear_embedding: nn.Module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=(patch_size, patch_size),
                                                     stride=(patch_size, patch_size))
        # Init layer normalization
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input image of the shape (B, C_in, H, W)

        Returns:
            embedding (torch.Tensor): Embedding of the shape (B, C_out, H / patch size, W / patch size)
        """
        # Perform linear embedding
        embedding: torch.Tensor = self.linear_embedding(input)
        # Perform normalization
        embedding: torch.Tensor = bhwc_to_bchw(self.normalization(bchw_to_bhwc(embedding)))
        return embedding


class SwinTransformerStage(nn.Module):
    r""" This class implements a stage of the Swin transformer including multiple layers.

    Args:
        in_channels (int): Number of input channels
        depth (int): Depth of the stage (number of layers)
        downscale (bool): If true input is downsampled (see Fig. 3 or V1 paper)
        input_resolution (Tuple[int, int]): Input resolution
        number_of_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        shift_size (int): Shifting size to be used
        ff_feature_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        dropout (float): Dropout in input mapping
        dropout_attention (float): Dropout rate of attention map
        dropout_path (float): Dropout in main path
        use_checkpoint (bool): If true checkpointing is utilized
        sequential_self_attention (bool): If true sequential self-attention is performed
        use_deformable_block (bool): If true deformable block is used
    """

    def __init__(self,
                 in_channels: int,
                 depth: int,
                 downscale: bool,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: Union[List[float], float] = 0.0,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 use_checkpoint: bool = False,
                 sequential_self_attention: bool = False,
                 use_deformable_block: bool = False) -> None:
        # Call super constructor
        super(SwinTransformerStage, self).__init__()
        # Save parameters
        self.use_checkpoint: bool = use_checkpoint
        self.downscale: bool = downscale
        # Init downsampling
        self.downsample: nn.Module = PatchMerging(in_channels=in_channels) if downscale else nn.Identity()
        # Update resolution and channels
        self.input_resolution: Tuple[int, int] = (input_resolution[0] // 2, input_resolution[1] // 2) \
            if downscale else input_resolution
        in_channels = in_channels * 2 if downscale else in_channels
        # Get block
        block = DeformableSwinTransformerBlock if use_deformable_block else SwinTransformerBlock
        # Init blocks
        self.blocks: nn.ModuleList = nn.ModuleList([
            block(in_channels=in_channels,
                  input_resolution=self.input_resolution,
                  number_of_heads=number_of_heads,
                  window_size=window_size,
                  shift_size=0 if ((index % 2) == 0) else window_size // 2,
                  ff_feature_ratio=ff_feature_ratio,
                  dropout=dropout,
                  dropout_attention=dropout_attention,
                  dropout_path=dropout_path[index] if isinstance(dropout_path, list) else dropout_path,
                  sequential_self_attention=sequential_self_attention)
            for index in range(depth)])

    def update_resolution(self,
                          new_window_size: int,
                          new_input_resolution: Tuple[int, int]) -> None:
        """ Method updates the resolution to utilize and the window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_input_resolution (Tuple[int, int]): New input resolution
        """
        # Update resolution
        self.input_resolution: Tuple[int, int] = (new_input_resolution[0] // 2, new_input_resolution[1] // 2) \
            if self.downscale else new_input_resolution
        # Update resolution of each block
        for block in self.blocks:  # type: SwinTransformerBlock
            block.update_resolution(new_window_size=new_window_size, new_input_resolution=self.input_resolution)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        # Downscale input tensor
        output: torch.Tensor = self.downsample(input)
        # Forward pass of each block
        for block in self.blocks:  # type: nn.Module
            # Perform checkpointing if utilized
            if self.use_checkpoint:
                output: torch.Tensor = checkpoint.checkpoint(block, output)
            else:
                output: torch.Tensor = block(output)
        return output


class SwinTransformerV2CR(nn.Module):
    r""" Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`  -
          https://arxiv.org/pdf/2111.09883

    Args:
        img_size (Tuple[int, int]): Input resolution.
        in_chans (int): Number of input channels.
        depths (int): Depth of the stage (number of layers).
        num_heads (int): Number of attention heads to be utilized.
        embed_dim (int): Patch embedding dimension. Default: 96
        num_classes (int): Number of output classes. Default: 1000
        window_size (int): Window size to be utilized. Default: 7
        patch_size (int | tuple(int)): Patch size. Default: 4
        mlp_ratio (int):  Ratio of the hidden dimension in the FFN to the input channels. Default: 4
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Dropout rate of attention map. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        use_checkpoint (bool): If true checkpointing is utilized. Default: False
        sequential_self_attention (bool): If true sequential self-attention is performed. Default: False
        use_deformable_block (bool): If true deformable block is used. Default: False
    """

    def __init__(self,
                 img_size: Tuple[int, int],
                 in_chans: int,
                 depths: Tuple[int, ...],
                 num_heads: Tuple[int, ...],
                 embed_dim: int = 96,
                 num_classes: int = 1000,
                 window_size: int = 7,
                 patch_size: int = 4,
                 mlp_ratio: int = 4,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 use_checkpoint: bool = False,
                 sequential_self_attention: bool = False,
                 use_deformable_block: bool = False,
                 **kwargs: Any) -> None:
        # Call super constructor
        super(SwinTransformerV2CR, self).__init__()
        # Save parameters
        self.num_classes: int = num_classes
        self.patch_size: int = patch_size
        self.input_resolution: Tuple[int, int] = img_size
        self.window_size: int = window_size
        self.num_features: int = int(embed_dim * (2 ** len(depths) - 1))
        # Init patch embedding
        self.patch_embedding: nn.Module = PatchEmbedding(in_channels=in_chans, out_channels=embed_dim,
                                                         patch_size=patch_size)
        # Compute patch resolution
        patch_resolution: Tuple[int, int] = (img_size[0] // patch_size, img_size[1] // patch_size)
        # Path dropout dependent on depth
        drop_path_rate = torch.linspace(0., drop_path_rate, sum(depths)).tolist()
        # Init stages
        self.stages: nn.ModuleList = nn.ModuleList()
        for index, (depth, number_of_head) in enumerate(zip(depths, num_heads)):
            self.stages.append(
                SwinTransformerStage(
                    in_channels=embed_dim * (2 ** max(index - 1, 0)),
                    depth=depth,
                    downscale=index != 0,
                    input_resolution=(patch_resolution[0] // (2 ** max(index - 1, 0)),
                                      patch_resolution[1] // (2 ** max(index - 1, 0))),
                    number_of_heads=number_of_head,
                    window_size=window_size,
                    ff_feature_ratio=mlp_ratio,
                    dropout=drop_rate,
                    dropout_attention=attn_drop_rate,
                    dropout_path=drop_path_rate[sum(depths[:index]):sum(depths[:index + 1])],
                    use_checkpoint=use_checkpoint,
                    sequential_self_attention=sequential_self_attention,
                    use_deformable_block=use_deformable_block and (index > 0)
                ))
        # Init final adaptive average pooling, and classification head
        self.average_pool: nn.Module = nn.AdaptiveAvgPool2d(1)
        self.head: nn.Module = nn.Linear(in_features=self.num_features,
                                         out_features=num_classes)

    def update_resolution(self,
                          new_input_resolution: Optional[Tuple[int, int]] = None,
                          new_window_size: Optional[int] = None) -> None:
        """ Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            new_window_size (Optional[int]): New window size if None current window size is used
            new_input_resolution (Optional[Tuple[int, int]]): New input resolution if None current resolution is used
        """
        # Check parameters
        if new_input_resolution is None:
            new_input_resolution = self.input_resolution
        if new_window_size is None:
            new_window_size = self.window_size
        # Compute new patch resolution
        new_patch_resolution: Tuple[int, int] = (new_input_resolution[0] // self.patch_size,
                                                 new_input_resolution[1] // self.patch_size)
        # Update resolution of each stage
        for index, stage in enumerate(self.stages):  # type: int, SwinTransformerStage
            stage.update_resolution(new_window_size=new_window_size,
                                    new_input_resolution=(new_patch_resolution[0] // (2 ** max(index - 1, 0)),
                                                          new_patch_resolution[1] // (2 ** max(index - 1, 0))))

    def get_classifier(self) -> nn.Module:
        """ Method returns the classification head of the model.
        Returns:
            head (nn.Module): Current classification head
        """
        head: nn.Module = self.head
        return head

    def reset_classifier(self, num_classes: int, global_pool: str = '') -> None:
        """ Method results the classification head

        Args:
            num_classes (int): Number of classes to be predicted
            global_pool (str): Unused
        """
        self.num_classes: int = num_classes
        self.head: nn.Module = nn.Linear(in_features=self.num_features, out_features=num_classes) \
            if num_classes > 0 else nn.Identity()

    def forward_features(self,
                         input: torch.Tensor) -> List[torch.Tensor]:
        """ Forward pass to extract feature maps of each stage.

        Args:
            input (torch.Tensor): Input images of the shape (B, C, H, W)

        Returns:
            features (List[torch.Tensor]): List of feature maps from each stage
        """
        # Check input resolution
        assert input.shape[2:] == self.input_resolution, \
            "Input resolution and utilized resolution does not match. Please update the models resolution by calling " \
            "update_resolution the provided method."
        # Perform patch embedding
        output: torch.Tensor = self.patch_embedding(input)
        # Init list to store feature
        features: List[torch.Tensor] = []
        # Forward pass of each stage
        for stage in self.stages:
            output: torch.Tensor = stage(output)
            features.append(output)
        return features

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input images of the shape (B, C, H, W)

        Returns:
            classification (torch.Tensor): Classification of the shape (B, num_classes)
        """
        # Check input resolution
        assert input.shape[2:] == self.input_resolution, \
            "Input resolution and utilized resolution does not match. Please update the models resolution by calling " \
            "update_resolution the provided method."
        # Perform patch embedding
        output: torch.Tensor = self.patch_embedding(input)
        # Forward pass of each stage
        for stage in self.stages:
            output: torch.Tensor = stage(output)
        # Perform average pooling
        output: torch.Tensor = self.average_pool(output)
        # Predict classification
        classification: torch.Tensor = self.head(output)
        return classification


def _create_swin_transformer_v2_cr(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        SwinTransformerV2CR, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model


@register_model
def swin_v2_cr_tiny_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-T V2 CR @ 384x384, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(384, 384), patch_size=4, window_size=12, embed_dim=96, depths=(2, 2, 6, 2),
                        num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_tiny_patch4_window12_384', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_tiny_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-T V2 CR @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(224, 224), patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2),
                        num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_tiny_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_small_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-S V2 CR @ 384x384, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(384, 384), patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2),
                        num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_small_patch4_window12_384', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_small_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-S V2 CR @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(224, 224), patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2),
                        num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_small_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_base_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-B V2 CR @ 384x384, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(384, 384), patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2),
                        num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_base_patch4_window12_384', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_base_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-B V2 CR @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(224, 224), patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                        num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_base_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_large_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-L V2 CR @ 384x384, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(384, 384), patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2),
                        num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_large_patch4_window12_384', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_large_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-L V2 CR @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(224, 224), patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2),
                        num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_large_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_huge_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-H V2 CR @ 384x384, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(384, 384), patch_size=4, window_size=12, embed_dim=352, depths=(2, 2, 18, 2),
                        num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_huge_patch4_window12_384', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_huge_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-H V2 CR @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(224, 224), patch_size=4, window_size=7, embed_dim=352, depths=(2, 2, 18, 2),
                        num_heads=(11, 22, 44, 88), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_huge_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_giant_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-G V2 CR @ 384x384, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(384, 384), patch_size=4, window_size=12, embed_dim=512, depths=(2, 2, 18, 2),
                        num_heads=(16, 32, 64, 128), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_giant_patch4_window12_384', pretrained=pretrained, **model_kwargs)


@register_model
def swin_v2_cr_giant_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-G V2 CR @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(img_size=(224, 224), patch_size=4, window_size=7, embed_dim=512, depths=(2, 2, 42, 2),
                        num_heads=(16, 32, 64, 128), **kwargs)
    return _create_swin_transformer_v2_cr('swin_v2_cr_giant_patch4_window7_224', pretrained=pretrained, **model_kwargs)


if __name__ == '__main__':
    model = swin_v2_cr_tiny_patch4_window12_384(pretrained=False)
    model = swin_v2_cr_tiny_patch4_window7_224(pretrained=False)

    model = swin_v2_cr_small_patch4_window12_384(pretrained=False)
    model = swin_v2_cr_small_patch4_window7_224(pretrained=False)

    model = swin_v2_cr_base_patch4_window12_384(pretrained=False)
    model = swin_v2_cr_base_patch4_window7_224(pretrained=False)

    model = swin_v2_cr_large_patch4_window12_384(pretrained=False)
    model = swin_v2_cr_large_patch4_window7_224(pretrained=False)
