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
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_kernel_midpoint_mask(
        kernel: Tuple[int, int],
        *,
        inplace_mask = None,
        shape: Optional[Tuple[int, int]] = None,
        device = None,
        dtype = None,
):
    """Build a mask of kernel midpoints.

    This predicts the kernel midpoints that conv2d (and related kernel functions)
    would place a kernel.

    The *midpoint* of a kernel is computed as ``size / 2``:
    * the midpoint of odd kernels is the middle: `mid(3) == 1`
    * the midpoint of even kernels is the first point in the second half: `mid(4) == 2`

    Requires `kernel <= min(h, w)`.

    When an `inplace_mask` is not provided, a new mask of `1`s is allocated,
    and then the `0` locations are cleared.

    When an `inplace_mask` is provided, the `0` locations are cleared on the mask,
    and no other changes are made. `shape`, `dtype`, and `device` must match, if
    they are provided.

    Args:
        kernel: the (kh, hw) shape of the kernel.
        inplace_mask: if supplied, updates will apply to the inplace_mask,
          and device and dtype will be ignored. Only clears 'false' locations.
        shape: the (h, w) shape of the tensor.
        device: the target device.
        dtype: the target dtype.

    Returns:
        a (h, w) bool mask tensor.
    """
    if inplace_mask is not None:
        mask = inplace_mask

        if shape:
            assert shape == mask.shape[-2], f"{shape=} !~= {mask.shape=}"

        shape = mask.shape

        if device:
            device = torch.device(device)
            assert device == mask.device, f"{device=} != {mask.device=}"

        if dtype:
            dtype = torch.dtype(dtype)
            assert dtype == inplace_mask.dtype, f"{dtype=} != {mask.dtype=}"

    else:
        mask = torch.ones(shape, dtype=dtype, device=device)

    h, w = shape
    kh, kw = kernel
    assert kh <= h and kw <= w, f"{kernel=} ! <= {shape=}"

    # Set to 0, rather than set to 1, so we can clear the inplace mask.
    mask[:kh // 2, :] = 0
    mask[h - (kh - 1) // 2:, :] = 0
    mask[:, :kw // 2] = 0
    mask[:, w - (kw - 1) // 2:] = 0

    return mask


def drop_block_2d(
        x,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0,
        with_noise: bool = False,
        inplace: bool = False,
        batchwise: bool = False,
        messy: bool = False,
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
        messy: partial-blocks at the edges, faster.

    Returns:
        If inplace, the modified `x`; otherwise, the dropped copy of `x`, on the same device.
    """
    B, C, H, W = x.shape
    total_size = W * H

    # TODO: This behaves oddly when clipped_block_size < block_size.
    clipped_block_size = min(block_size, H, W)

    gamma = (
        float(gamma_scale * drop_prob * total_size)
        / float(clipped_block_size ** 2)
        / float((H - block_size + 1) * (W - block_size + 1))
    )

    # batchwise => one mask for whole batch, quite a bit faster
    mask_shape = (1 if batchwise else B, C, H, W)

    block_mask = torch.empty(
        mask_shape,
        dtype=x.dtype,
        device=x.device
    ).bernoulli_(gamma)

    if not messy:
        conv2d_kernel_midpoint_mask(
            kernel=(clipped_block_size, clipped_block_size),
            inplace_mask=block_mask,
        )

    block_mask = F.max_pool2d(
        block_mask,
        kernel_size=clipped_block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if inplace:
        x.mul_(block_mask)
    else:
        x = x * block_mask

    # From this point on, we do inplace ops on X.

    if with_noise:
        noise = torch.randn(mask_shape, dtype=x.dtype, device=x.device)
        # x += (noise * (1 - block_mask))
        block_mask.neg_().add_(1)
        noise.mul_(block_mask)
        x.add_(noise)

    else:
        # x *= (size(block_mask) / sum(block_mask))
        total = block_mask.to(dtype=torch.float32).sum()
        normalize_scale = block_mask.numel() / total.add(1e-7).to(x.dtype)
        x.mul_(normalize_scale)

    return x


def drop_block_fast_2d(
        x: torch.Tensor,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0,
        with_noise: bool = False,
        inplace: bool = False,
):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

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
        messy=True,
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
        messy: partial-blocks at the edges, faster.
    """
    drop_prob: float
    block_size: int
    gamma_scale: float
    with_noise: bool
    inplace: bool
    batchwise: bool
    messy: bool

    def __init__(
            self,
            drop_prob: float = 0.1,
            block_size: int = 7,
            gamma_scale: float = 1.0,
            with_noise: bool = False,
            inplace: bool = False,
            batchwise: bool = False,
            messy: bool = True,
    ):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.messy = messy

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
            messy=self.messy)


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
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
