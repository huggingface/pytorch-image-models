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
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    if new_size == old_size:  # might not both be same container type
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    if verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        print(posemb_prefix.shape, posemb.shape)
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb
