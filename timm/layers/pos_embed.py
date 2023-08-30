""" Position Embedding Utilities

Hacked together by / Copyright 2022 Ross Wightman
"""
import logging
import math
from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F

from .helpers import to_2tuple

_logger = logging.getLogger(__name__)


def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb


def resample_abs_pos_embed_nhwc(
        posemb,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    if new_size[0] == posemb.shape[-3] and new_size[1] == posemb.shape[-2]:
        return posemb

    orig_dtype = posemb.dtype
    posemb = posemb.float()
    # do the interpolation
    posemb = posemb.reshape(1, posemb.shape[-3], posemb.shape[-2], posemb.shape[-1]).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).to(orig_dtype)

    if not torch.jit.is_scripting() and verbose:
        _logger.info(f'Resized position embedding: {posemb.shape[-3:-1]} to {new_size}.')

    return posemb


def resample_relative_position_bias_table(
        position_bias_table,
        new_size,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False
):
    """
    Resample relative position bias table suggested in LeVit
    Adapted from: https://github.com/microsoft/Cream/blob/main/TinyViT/utils.py
    """
    L1, nH1 = position_bias_table.size()
    L2, nH2 = new_size
    assert nH1 == nH2
    if L1 != L2:
        orig_dtype = position_bias_table.dtype
        position_bias_table = position_bias_table.float()
        # bicubic interpolate relative_position_bias_table if not match
        S1 = int(L1 ** 0.5)
        S2 = int(L2 ** 0.5)
        relative_position_bias_table_resized = F.interpolate(
            position_bias_table.permute(1, 0).view(1, nH1, S1, S1),
            size=(S2, S2),
            mode=interpolation,
            antialias=antialias)
        relative_position_bias_table_resized = \
            relative_position_bias_table_resized.view(nH2, L2).permute(1, 0)
        relative_position_bias_table_resized.to(orig_dtype)
        if not torch.jit.is_scripting() and verbose:
            _logger.info(f'Resized position bias: {L1, nH1} to {L2, nH2}.')
        return relative_position_bias_table_resized
    else:
        return position_bias_table
