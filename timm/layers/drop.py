""" DropBlock, DropPath

PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

Papers:
DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

Code:
DropBlock impl inspired by two Tensorflow impl that I liked:
 - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
 - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_block_2d(
        x: torch.Tensor,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0,
        with_noise: bool = False,
        inplace: bool = False,
        couple_channels: bool = True,
        scale_by_keep: bool = True,
):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option.

    Args:
        x: Input tensor of shape (B, C, H, W).
        drop_prob: Probability of dropping a block.
        block_size: Size of the block to drop.
        gamma_scale: Scale factor for the drop probability.
        with_noise: If True, add gaussian noise to dropped regions instead of zeros.
        inplace: If True, perform operation in-place.
        couple_channels: If True, all channels share the same drop mask (per the original paper).
            If False, each channel gets an independent mask.
        scale_by_keep: If True, scale kept activations to maintain expected values.

    Returns:
        Tensor with dropped blocks, same shape as input.
    """
    B, C, H, W = x.shape
    kh, kw = min(block_size, H), min(block_size, W)

    # Compute gamma (seed drop rate) - probability of dropping each spatial location
    gamma = float(gamma_scale * drop_prob * H * W) / float(kh * kw) / float((H - kh + 1) * (W - kw + 1))

    # Generate drop mask: 1 at block centers to drop, 0 elsewhere
    # couple_channels=True means all channels share same spatial mask (matches paper)
    noise_shape = (B, 1 if couple_channels else C, H, W)
    with torch.no_grad():
        block_mask = torch.empty(noise_shape, dtype=x.dtype, device=x.device).bernoulli_(gamma)

        # Expand block centers to full blocks using max pooling
        block_mask = F.max_pool2d(
            block_mask,
            kernel_size=(kh, kw),
            stride=1,
            padding=(kh // 2, kw // 2),
        )
        # Handle even kernel sizes - max_pool2d output is 1 larger in each even dimension
        if kh % 2 == 0 or kw % 2 == 0:
            # Fix for even kernels proposed by https://github.com/crutcher
            block_mask = block_mask[..., (kh + 1) % 2:, (kw + 1) % 2:]

        keep_mask = 1. - block_mask

    if with_noise:
        with torch.no_grad():
            noise = torch.empty_like(keep_mask).normal_()
            noise.mul_(block_mask)

        if inplace:
            x.mul_(keep_mask).add_(noise)
        else:
            x = x * keep_mask + noise
    else:
        if scale_by_keep:
            with torch.no_grad():
                # Normalize to maintain expected values (scale up kept activations)
                normalize_scale = keep_mask.numel() / keep_mask.to(dtype=torch.float32).sum().add(1e-7)
                keep_mask.mul_(normalize_scale.to(x.dtype))

        if inplace:
            x.mul_(keep_mask)
        else:
            x = x * keep_mask

    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    Args:
        drop_prob: Probability of dropping a block.
        block_size: Size of the block to drop.
        gamma_scale: Scale factor for the drop probability.
        with_noise: If True, add gaussian noise to dropped regions instead of zeros.
        inplace: If True, perform operation in-place.
        couple_channels: If True, all channels share the same drop mask (per the original paper).
            If False, each channel gets an independent mask.
        scale_by_keep: If True, scale kept activations to maintain expected values.
    """

    def __init__(
            self,
            drop_prob: float = 0.1,
            block_size: int = 7,
            gamma_scale: float = 1.0,
            with_noise: bool = False,
            inplace: bool = False,
            couple_channels: bool = True,
            scale_by_keep: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.couple_channels = couple_channels
        self.scale_by_keep = scale_by_keep

        # Backwards compatibility: silently consume args removed in v1.0.23, warn on unknown
        deprecated_args = {'batchwise', 'fast'}
        for k in kwargs:
            if k not in deprecated_args:
                import warnings
                warnings.warn(f"DropBlock2d() got unexpected keyword argument '{k}'")

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        return drop_block_2d(
            x,
            drop_prob=self.drop_prob,
            block_size=self.block_size,
            gamma_scale=self.gamma_scale,
            with_noise=self.with_noise,
            inplace=self.inplace,
            couple_channels=self.couple_channels,
            scale_by_keep=self.scale_by_keep,
        )


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def calculate_drop_path_rates(
        drop_path_rate: float,
        depths: Union[int, List[int]],
        stagewise: bool = False,
) -> Union[List[float], List[List[float]]]:
    """Generate drop path rates for stochastic depth.

    This function handles two common patterns for drop path rate scheduling:
    1. Per-block: Linear increase from 0 to drop_path_rate across all blocks
    2. Stage-wise: Linear increase across stages, with same rate within each stage

    Args:
        drop_path_rate: Maximum drop path rate (at the end).
        depths: Either a single int for total depth (per-block mode) or
                list of ints for depths per stage (stage-wise mode).
        stagewise: If True, use stage-wise pattern. If False, use per-block pattern.
                   When depths is a list, stagewise defaults to True.

    Returns:
        For per-block mode: List of drop rates, one per block.
        For stage-wise mode: List of lists, drop rates per stage.
    """
    if isinstance(depths, int):
        # Single depth value - per-block pattern
        if stagewise:
            raise ValueError("stagewise=True requires depths to be a list of stage depths")
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths, device='cpu')]
        return dpr
    else:
        # List of depths - can be either pattern
        total_depth = sum(depths)
        if stagewise:
            # Stage-wise pattern: same drop rate within each stage
            dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, total_depth, device='cpu').split(depths)]
            return dpr
        else:
            # Per-block pattern across all stages
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth, device='cpu')]
            return dpr
