"""DropBlock, DropPath

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

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_kernel_midpoint_mask(
    *,
    shape: Tuple[int, int],
    kernel: Tuple[int, int],
    device,
    dtype,
):
    """Build a mask of kernel midpoints.

    This predicts the kernel midpoints that conv2d (and related kernel functions)
    would place a kernel.

    The *midpoint* of a kernel is computed as ``size / 2``:
    * the midpoint of odd kernels is the middle: `mid(3) == 1`
    * the midpoint of even kernels is the first point in the second half: `mid(4) == 2`

    Requires `kernel <= min(h, w)`.

    A new mask of `1`s is allocated, and then the `0` locations are cleared.

    Args:
        kernel: the (kh, hw) shape of the kernel.
        shape: the (h, w) shape of the tensor.
        device: the target device.
        dtype: the target dtype.

    Returns:
        a (h, w) bool mask tensor.
    """
    h, w = shape
    kh, kw = kernel
    assert kh <= h and kw <= w, f"{kernel=} ! <= {shape=}"

    mask = torch.zeros(shape, dtype=dtype, device=device)

    mask[kh // 2 : h - ((kh - 1) // 2), kw // 2 : w - ((kw - 1) // 2)] = 1.0

    return mask


def drop_block_2d_drop_filter_(
    *, selection, kernel: Tuple[int, int], partial_edge_blocks: bool
):
    """Convert drop block gamma noise to a drop filter.

    This is a deterministic internal component of drop_block_2d.

    Args:
        selection: 4D (B, C, H, W) input selection noise;
          `1.0` at the midpoints of selected blocks to drop,
          `0.0` everywhere else. Expected to be gamma noise.
        kernel: the shape of the 2d kernel.
        partial_edge_blocks: permit partial blocks at the edges, faster.

    Returns:
        A drop filter, `1.0` at points to drop, `0.0` at points to keep.
    """

    if not partial_edge_blocks:
        selection = selection * conv2d_kernel_midpoint_mask(
            shape=selection.shape[-2:],
            kernel=kernel,
            dtype=selection.dtype,
            device=selection.device,
        )

    kh, kw = kernel

    drop_filter = F.max_pool2d(
        selection,
        kernel_size=kernel,
        stride=1,
        padding=[kh // 2, kw // 2],
    )
    if (kh % 2 == 0) or (kw % 2 == 0):
        drop_filter = drop_filter[..., (kh % 2 == 0) :, (kw % 2 == 0) :]

    return drop_filter


def drop_block_2d(
    x,
    drop_prob: float = 0.1,
    block_size: int = 7,
    gamma_scale: float = 1.0,
    with_noise: bool = False,
    inplace: bool = False,
    batchwise: bool = False,
    partial_edge_blocks: bool = False,
):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.

    Args:
        drop_prob: the probability of dropping any given block.
        block_size: the size of the dropped blocks; should be odd.
        gamma_scale: adjustment scale for the drop_prob.
        with_noise: should normal noise be added to the dropped region?
        inplace: if the drop should be applied in-place on the input tensor.
        batchwise: should the entire batch use the same drop mask?
        partial_edge_blocks: partial-blocks at the edges, faster.

    Returns:
        If inplace, the modified `x`; otherwise, the dropped copy of `x`, on the same device.
    """
    B, C, H, W = x.shape

    # TODO: This behaves oddly when clipped_block_size < block_size.
    kh = kw = block_size

    kernel = [min(kh, H), min(kw, W)]
    kh, kw = kernel

    gamma = (
        float(gamma_scale * drop_prob * H * W)
        / float(kh * kw)
        / float((H - kh + 1) * (W - kw + 1))
    )

    # batchwise => one mask for whole batch, quite a bit faster
    mask_shape = (1 if batchwise else B, C, H, W)

    selection = torch.empty(mask_shape, dtype=x.dtype, device=x.device).bernoulli_(
        gamma
    )

    drop_filter = drop_block_2d_drop_filter_(
        selection=selection,
        kernel=kernel,
        partial_edge_blocks=partial_edge_blocks,
    )
    keep_filter = 1.0 - drop_filter

    if inplace:
        x.mul_(keep_filter)
    else:
        x = x * keep_filter

    if with_noise:
        # x += (noise * (1 - block_mask))
        noise = torch.randn(mask_shape, dtype=x.dtype, device=x.device)

        if inplace:
            noise.mul_(drop_filter)
            x.add_(noise)
        else:
            x = x + noise * drop_filter

    else:
        # x *= (size(block_mask) / sum(block_mask))
        count = keep_filter.numel()
        total = keep_filter.to(dtype=torch.float32).sum()
        normalize_scale = count / total.add(1e-7).to(x.dtype)

        if inplace:
            x.mul_(normalize_scale)
        else:
            x = x * normalize_scale

    return x


def drop_block_fast_2d(
    x: torch.Tensor,
    drop_prob: float = 0.1,
    block_size: int = 7,
    gamma_scale: float = 1.0,
    with_noise: bool = False,
    inplace: bool = False,
):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    drop_block_2d(
        x=x,
        drop_prob=drop_prob,
        block_size=block_size,
        gamma_scale=gamma_scale,
        with_noise=with_noise,
        inplace=inplace,
        batchwise=True,
        partial_edge_blocks=True,
    )


class DropBlock2d(nn.Module):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    Args:
        drop_prob: the probability of dropping any given block.
        block_size: the size of the dropped blocks; should be odd.
        gamma_scale: adjustment scale for the drop_prob.
        with_noise: should normal noise be added to the dropped region?
        inplace: if the drop should be applied in-place on the input tensor.
        batchwise: should the entire batch use the same drop mask?
        partial_edge_blocks: partial-blocks at the edges, faster.
    """

    drop_prob: float
    block_size: int
    gamma_scale: float
    with_noise: bool
    inplace: bool
    batchwise: bool
    partial_edge_blocks: bool

    def __init__(
        self,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0,
        with_noise: bool = False,
        inplace: bool = False,
        batchwise: bool = False,
        partial_edge_blocks: bool = True,
    ):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.partial_edge_blocks = partial_edge_blocks

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x

        return drop_block_2d(
            x=x,
            drop_prob=self.drop_prob,
            block_size=self.block_size,
            gamma_scale=self.gamma_scale,
            with_noise=self.with_noise,
            inplace=self.inplace,
            batchwise=self.batchwise,
            partial_edge_blocks=self.partial_edge_blocks,
        )


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"
