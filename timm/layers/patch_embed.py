""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on code in:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
from typing import List, Optional, Callable

import torch
from torch import nn as nn
import torch.nn.functional as F

from .format import Format, nchw_to
from .helpers import to_2tuple
from .trace_utils import _assert

_logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        import functorch
        vmap = functorch.vmap
    except ImportError:
        if hasattr(torch, 'vmap'):
            vmap = torch.vmap
        else:
            assert False, "functorch or a version of torch with vmap is required for FlexiViT resizing."

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    if verbose:
        _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    return v_resample_kernel(patch_embed)


# def divs(n, m=None):
#     m = m or n // 2
#     if m == 1:
#         return [1]
#     if n % m == 0:
#         return [m] + divs(n, m - 1)
#     return divs(n, m - 1)
#
#
# class FlexiPatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding w/ Flexible Patch sizes (FlexiViT)
#     FIXME WIP
#     """
#     def __init__(
#             self,
#             img_size=240,
#             patch_size=16,
#             in_chans=3,
#             embed_dim=768,
#             base_img_size=240,
#             base_patch_size=32,
#             norm_layer=None,
#             flatten=True,
#             bias=True,
#     ):
#         super().__init__()
#         self.img_size = to_2tuple(img_size)
#         self.patch_size = to_2tuple(patch_size)
#         self.num_patches = 0
#
#         # full range for 240 = (5, 6, 8, 10, 12, 14, 15, 16, 20, 24, 30, 40, 48)
#         self.seqhw = (6, 8, 10, 12, 14, 15, 16, 20, 24, 30)
#
#         self.base_img_size = to_2tuple(base_img_size)
#         self.base_patch_size = to_2tuple(base_patch_size)
#         self.base_grid_size = tuple([i // p for i, p in zip(self.base_img_size, self.base_patch_size)])
#         self.base_num_patches = self.base_grid_size[0] * self.base_grid_size[1]
#
#         self.flatten = flatten
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         if self.patch_size == self.base_patch_size:
#             weight = self.proj.weight
#         else:
#             weight = resample_patch_embed(self.proj.weight, self.patch_size)
#         patch_size = self.patch_size
#         x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x
