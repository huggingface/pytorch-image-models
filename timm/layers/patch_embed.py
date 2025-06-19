""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on code in:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv2d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size, verbose=True))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


class PatchEmbedWithSize(PatchEmbed):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
        )

    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(H % self.patch_size[0] == 0, f"Input image height ({H}) must be divisible by patch size ({self.patch_size[0]}).")
            _assert(W % self.patch_size[1] == 0, f"Input image width ({W}) must be divisible by patch size ({self.patch_size[1]}).")

        x = self.proj(x)
        feat_size = x.shape[-2:]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x, feat_size


# FIXME to remove, keeping for comparison for now
def resample_patch_embed_old(
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
        from torch import vmap
    except ImportError:
        from functorch import vmap

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
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


DTYPE_INTERMEDIATE = torch.float32


def _compute_resize_matrix(
    old_size: Tuple[int, int],
    new_size: Tuple[int, int],
    interpolation: str,
    antialias: bool,
    device: torch.device,
    dtype: torch.dtype = DTYPE_INTERMEDIATE
) -> torch.Tensor:
    """Computes the resize matrix basis vectors and interpolates them to new_size."""
    old_h, old_w = old_size
    new_h, new_w = new_size
    old_total = old_h * old_w
    new_total = new_h * new_w

    eye_matrix = torch.eye(old_total, device=device, dtype=dtype)
    basis_vectors_batch = eye_matrix.reshape(old_total, 1, old_h, old_w)
    resized_basis_vectors_batch = F.interpolate(
        basis_vectors_batch,
        size=new_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False
    ) # Output shape: (old_total, 1, new_h, new_w)
    resize_matrix = resized_basis_vectors_batch.squeeze(1).permute(1, 2, 0).reshape(new_total, old_total)
    return resize_matrix # Shape: (new_total, old_total)


def _apply_resampling(
    patch_embed: torch.Tensor,
    pinv_matrix: torch.Tensor,
    new_size_tuple: Tuple[int, int],
    orig_dtype: torch.dtype,
    intermediate_dtype: torch.dtype = DTYPE_INTERMEDIATE
) -> torch.Tensor:
    """ Simplified resampling w/o vmap use.
    As proposed by https://github.com/stas-sl
    """
    c_out, c_in, *_ = patch_embed.shape
    patch_embed = patch_embed.reshape(c_out, c_in, -1).to(dtype=intermediate_dtype)
    pinv_matrix = pinv_matrix.to(dtype=intermediate_dtype)
    resampled_patch_embed = patch_embed @ pinv_matrix  # (C_out, C_in, P_old * P_old) @ (P_old * P_old, P_new * P_new)
    resampled_patch_embed = resampled_patch_embed.reshape(c_out, c_in, *new_size_tuple).to(dtype=orig_dtype)
    return resampled_patch_embed


def resample_patch_embed(
        patch_embed: torch.Tensor,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """ Standalone function (computes matrix on each call). """
    assert len(patch_embed.shape) == 4, "Input tensor should be 4D (out_ch, in_ch, h, w)"
    assert len(new_size) == 2, "New shape should only be hw (height, width)"

    old_size_tuple: Tuple[int, int] = tuple(patch_embed.shape[-2:])
    new_size_tuple: Tuple[int, int] = tuple(new_size)

    if old_size_tuple == new_size_tuple:
        return patch_embed

    device = patch_embed.device
    orig_dtype = patch_embed.dtype

    resize_mat = _compute_resize_matrix(
        old_size_tuple, new_size_tuple, interpolation, antialias, device, DTYPE_INTERMEDIATE
    )
    pinv_matrix = torch.linalg.pinv(resize_mat)  # Calculates the pseudoinverse matrix used for resampling
    resampled_patch_embed = _apply_resampling(
        patch_embed, pinv_matrix, new_size_tuple, orig_dtype, DTYPE_INTERMEDIATE
    )
    return resampled_patch_embed


class PatchEmbedResamplerFixedOrigSize(nn.Module):
    """
    Resample patch embedding weights from a fixed original size,
    caching the pseudoinverse matrix based on the target size.
    """
    def __init__(
        self,
        orig_size: Tuple[int, int],
        interpolation: str = 'bicubic',
        antialias: bool = True
    ):
        """
        Args:
            orig_size (Tuple[int, int]): The expected original (height, width) of input patch_embed tensors.
            interpolation (str): Interpolation mode.
            antialias (bool): Use anti-aliasing filter in resize.
        """
        super().__init__()
        assert isinstance(orig_size, tuple) and len(orig_size) == 2, \
            "`orig_size` must be a tuple of (height, width)"
        self.orig_size = orig_size # expected original size
        self.interpolation = interpolation
        self.antialias = antialias
        # Cache map key is the target new_size tuple
        self._pinv_cache_map: Dict[Tuple[int, int], str] = {}

    def _get_or_create_pinv_matrix(
        self,
        new_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = DTYPE_INTERMEDIATE
    ) -> torch.Tensor:
        """Retrieves the cached pinv matrix or computes and caches it for the given new_size."""
        cache_key = new_size
        buffer_name = self._pinv_cache_map.get(cache_key)

        if buffer_name and hasattr(self, buffer_name):
            pinv_matrix = getattr(self, buffer_name)
            if pinv_matrix.device == device and pinv_matrix.dtype == dtype:
                 return pinv_matrix

        # Calculate the matrix if not cached or needs update
        resize_mat = _compute_resize_matrix(
            self.orig_size, new_size, self.interpolation, self.antialias, device, dtype
        )
        pinv_matrix = torch.linalg.pinv(resize_mat)  # Calculates the pseudoinverse matrix used for resampling

        # Cache using register_buffer
        buffer_name = f"pinv_{new_size[0]}x{new_size[1]}"
        if hasattr(self, buffer_name):
             delattr(self, buffer_name)
        self.register_buffer(buffer_name, pinv_matrix)
        self._pinv_cache_map[cache_key] = buffer_name # Map new_size key to buffer name

        return pinv_matrix

    def forward(self, patch_embed: torch.Tensor, new_size: List[int]) -> torch.Tensor:
        """ Resamples the patch embedding weights to new_size.

        Args:
            patch_embed (torch.Tensor): Original weights (out_ch, in_ch, H_orig, W_orig).
            new_size (List[int]): Target [height, width].

        Returns:
            torch.Tensor: Resampled weights.
        """
        assert len(patch_embed.shape) == 4
        assert len(new_size) == 2

        # Input Validation
        input_size = tuple(patch_embed.shape[-2:])
        assert input_size == self.orig_size, \
            f"Input patch_embed spatial size {input_size} does not match " \
            f"module's expected original size {self.orig_size}"

        new_size_tuple: Tuple[int, int] = tuple(new_size)

        # Check no-op case against self.orig_size
        if self.orig_size == new_size_tuple:
            return patch_embed

        device = patch_embed.device
        orig_dtype = patch_embed.dtype

        # Get or compute the required pseudoinverse matrix
        pinv_matrix = self._get_or_create_pinv_matrix(new_size_tuple, device)

        # Apply the resampling
        resampled_patch_embed = _apply_resampling(patch_embed, pinv_matrix, new_size_tuple, orig_dtype)

        return resampled_patch_embed


class PatchEmbedInterpolator(nn.Module):
    """Dynamically interpolates patch embedding weights for variable patch sizes.

    This module wraps patch embedding weight resampling functionality to support
    on-the-fly patch size variation during training. It handles both Conv2d and
    Linear patch embeddings.

    Args:
        base_patch_size: The original patch size the model was initialized with
        in_chans: Number of input channels
        embed_dim: Embedding dimension
        interpolation: Interpolation mode for resampling
        antialias: Whether to use antialiasing during interpolation
    """

    def __init__(
        self,
        base_patch_size: Tuple[int, int],
        in_chans: int = 3,
        embed_dim: int = 768,
        interpolation: str = 'bicubic',
        antialias: bool = True,
    ):
        super().__init__()
        self.base_patch_size = base_patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.interpolation = interpolation
        self.antialias = antialias

    def resample_linear_weight(
        self,
        weight: torch.Tensor,
        target_patch_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Resample linear patch embedding weights for a new patch size.

        Args:
            weight: Linear weight tensor of shape [embed_dim, patch_h * patch_w * in_chans]
            target_patch_size: Target (patch_h, patch_w) to resample to

        Returns:
            Resampled weight tensor
        """
        if target_patch_size == self.base_patch_size:
            return weight

        embed_dim = weight.shape[0]
        base_ph, base_pw = self.base_patch_size
        target_ph, target_pw = target_patch_size

        # Reshape linear weight to conv2d format
        # [embed_dim, ph*pw*C] -> [embed_dim, C, ph, pw]
        weight_conv = weight.reshape(embed_dim, base_ph, base_pw, self.in_chans)
        weight_conv = weight_conv.permute(0, 3, 1, 2)

        # Resample using existing function
        weight_conv_resampled = resample_patch_embed(
            weight_conv,
            new_size=[target_ph, target_pw],
            interpolation=self.interpolation,
            antialias=self.antialias,
            verbose=False,
        )

        # Reshape back to linear format
        # [embed_dim, C, ph, pw] -> [embed_dim, ph*pw*C]
        weight_resampled = weight_conv_resampled.permute(0, 2, 3, 1)
        weight_resampled = weight_resampled.reshape(embed_dim, -1)

        return weight_resampled

    def resample_conv_weight(
        self,
        weight: torch.Tensor,
        target_patch_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Resample conv2d patch embedding weights for a new patch size.

        Args:
            weight: Conv2d weight tensor of shape [embed_dim, in_chans, patch_h, patch_w]
            target_patch_size: Target (patch_h, patch_w) to resample to

        Returns:
            Resampled weight tensor
        """
        if target_patch_size == self.base_patch_size:
            return weight

        # Resample using existing function
        weight_resampled = resample_patch_embed(
            weight,
            new_size=list(target_patch_size),
            interpolation=self.interpolation,
            antialias=self.antialias,
            verbose=False,
        )

        return weight_resampled

    def forward(
        self,
        patches: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: Optional[torch.Tensor] = None,
        patch_size: Optional[Tuple[int, int]] = None,
        is_linear: bool = True,
    ) -> torch.Tensor:
        """Apply patch embedding with dynamic weight resampling.

        Args:
            patches: Input patches
                - For linear mode with resampling: [B, N, Ph, Pw, C]
                - For linear mode without resampling: [B, N, Ph*Pw*C]
                - For conv mode: [B, C, H, W]
            proj_weight: Original projection weight
            proj_bias: Optional projection bias
            patch_size: Current patch size (if None, uses base_patch_size)
            is_linear: Whether using linear (True) or conv2d (False) projection

        Returns:
            Embedded patches
        """
        if patch_size is None:
            patch_size = self.base_patch_size

        if is_linear:
            if patch_size != self.base_patch_size:
                # Need to resample - expects unflattened patches
                assert patches.ndim == 5, "Patches must be [B, N, Ph, Pw, C] for resampling"
                B, N, Ph, Pw, C = patches.shape

                # Resample the weight
                weight_resampled = self.resample_linear_weight(proj_weight, patch_size)

                # Flatten patches and apply linear projection
                patches_flat = patches.reshape(B, N, -1)
                output = torch.nn.functional.linear(patches_flat, weight_resampled, proj_bias)
            else:
                # No resampling needed, patches can be pre-flattened
                if patches.ndim == 5:
                    B, N, Ph, Pw, C = patches.shape
                    patches = patches.reshape(B, N, -1)
                output = torch.nn.functional.linear(patches, proj_weight, proj_bias)
        else:
            # Conv mode
            if patch_size != self.base_patch_size:
                weight_resampled = self.resample_conv_weight(proj_weight, patch_size)
                output = torch.nn.functional.conv2d(
                    patches, weight_resampled, proj_bias,
                    stride=patch_size, padding=0
                )
            else:
                output = torch.nn.functional.conv2d(
                    patches, proj_weight, proj_bias,
                    stride=patch_size, padding=0
                )

        return output

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
