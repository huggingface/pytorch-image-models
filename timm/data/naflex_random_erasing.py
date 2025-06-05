"""Patch-level random erasing augmentation for NaFlex Vision Transformers.

This module implements random erasing specifically designed for patchified images,
operating at the patch granularity rather than pixel level. It supports two modes:
- 'patch': Randomly erases individual patches (speckle-like noise)
- 'region': Erases contiguous rectangular regions of patches (similar to original RandomErasing)

The implementation is coordinate-aware, respecting valid patch boundaries and supporting
variable patch sizes in NaFlex training.

Hacked together by / Copyright 2025, Ross Wightman, Hugging Face
"""

import random
import math
from typing import Optional, Union, Tuple

import torch


class PatchRandomErasing:
    """Random erasing for patchified images in NaFlex format.

    Supports two modes:
    1. 'patch': Simple mode that erases randomly selected valid patches
    2. 'region': Erases rectangular regions at patch granularity
    """

    def __init__(
            self,
            erase_prob: float = 0.5,
            patch_drop_prob: float = 0.0,
            min_count: int = 1,
            max_count: Optional[int] = None,
            min_area: float = 0.02,
            max_area: float = 1 / 3,
            min_aspect: float = 0.3,
            max_aspect: Optional[float] = None,
            mode: str = 'const',
            value: float = 0.,
            spatial_mode: str = 'region',
            num_splits: int = 0,
            device: Union[str, torch.device] = 'cuda',
    ) -> None:
        """Initialize PatchRandomErasing.

        Args:
            erase_prob: Probability that the Random Erasing operation will be performed.
            patch_drop_prob: Patch dropout probability. Remove random patches instead of erasing.
            min_count: Minimum number of erasing operations.
            max_count: Maximum number of erasing operations.
            min_area: Minimum percentage of valid patches/area to erase.
            max_area: Maximum percentage of valid patches/area to erase.
            min_aspect: Minimum aspect ratio of erased area (only used in 'region' mode).
            max_aspect: Maximum aspect ratio of erased area (only used in 'region' mode).
            mode: Patch content mode, one of 'const', 'rand', or 'pixel'.
            value: Constant value for 'const' mode.
            spatial_mode: Erasing strategy, one of 'patch' or 'region'.
            num_splits: Number of splits to apply erasing to (0 for all).
            device: Computation device.
        """
        self.erase_prob = erase_prob
        self.patch_drop_prob = patch_drop_prob
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.min_area = min_area
        self.max_area = max_area

        # Aspect ratio params (for region mode)
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        # Number of splits
        self.num_splits = num_splits
        self.device = device

        # Strategy mode
        self.spatial_mode = spatial_mode
        assert self.spatial_mode in ('patch', 'region')

        # Value generation mode flags
        self.erase_mode = mode.lower()
        assert self.erase_mode in ('rand', 'pixel', 'const')
        self.const_value = value
        self.unique_noise_per_patch = True

    def _get_values(
            self,
            shape: Union[Tuple[int, ...], torch.Size],
            value: Optional[torch.Tensor] = None,
            dtype: torch.dtype = torch.float32,
            device: Optional[Union[str, torch.device]] = None
    ) -> torch.Tensor:
        """Generate values for erased patches based on the specified mode.

        Args:
            shape: Shape of patches to erase.
            value: Value to use in const (or rand) mode.
            dtype: Data type to use.
            device: Device to use.

        Returns:
            Tensor with values for erasing patches.
        """
        device = device or self.device
        if self.erase_mode == 'pixel':
            # only mode with erase shape that includes pixels
            return torch.empty(shape, dtype=dtype, device=device).normal_()
        else:
            shape = (1, 1, shape[-1]) if len(shape) == 3 else (1, shape[-1])
            if self.erase_mode == 'const' or value is not None:
                erase_value = value or self.const_value
                if isinstance(erase_value, (int, float)):
                    values = torch.full(shape, erase_value, dtype=dtype, device=device)
                else:
                    erase_value = torch.tensor(erase_value, dtype=dtype, device=device)
                    values = torch.expand_copy(erase_value, shape)
            else:
                values = torch.empty(shape, dtype=dtype, device=device).normal_()
            return values

    def _drop_patches(
            self,
            patches: torch.Tensor,
            patch_coord: torch.Tensor,
            patch_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch Dropout.

        Fully drops patches from datastream. Only mode that saves compute BUT requires support
        for non-contiguous patches and associated patch coordinate and valid handling.

        Args:
            patches: Tensor of patches.
            patch_coord: Tensor of patch coordinates.
            patch_valid: Tensor indicating which patches are valid.

        Returns:
            Tuple of (patches, patch_coord, patch_valid) with some patches dropped.
        """
        # FIXME WIP, not completed. Downstream support in model needed for non-contiguous valid patches
        if random.random() > self.erase_prob:
            return

        # Get indices of valid patches
        valid_indices = torch.nonzero(patch_valid, as_tuple=True)[0].tolist()

        # Skip if no valid patches
        if not valid_indices:
            return patches, patch_coord, patch_valid

        num_valid = len(valid_indices)
        if self.patch_drop_prob:
            # patch dropout mode, completely remove dropped patches (FIXME needs downstream support in model)
            num_keep = max(1, int(num_valid * (1. - self.patch_drop_prob)))
            keep_indices = torch.argsort(torch.randn(1, num_valid, device=self.device), dim=-1)[:, :num_keep]
            # maintain patch order, possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
            patches = patches.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + patches.shape[2:]))

        return patches, patch_coord, patch_valid

    def _erase_patches(
            self,
            patches: torch.Tensor,
            patch_coord: torch.Tensor,
            patch_valid: torch.Tensor,
            patch_shape: torch.Size,
            dtype: torch.dtype = torch.float32,
    ) -> None:
        """Apply erasing by selecting individual patches randomly.

        The simplest mode, aligned on patch boundaries. Behaves similarly to speckle or 'sprinkles'
        noise augmentation at patch size.

        Args:
            patches: Tensor of patches to modify in-place.
            patch_coord: Tensor of patch coordinates.
            patch_valid: Tensor indicating which patches are valid.
            patch_shape: Shape of individual patches.
            dtype: Data type for generated values.
        """
        if random.random() > self.erase_prob:
            return

        # Get indices of valid patches
        valid_indices = torch.nonzero(patch_valid, as_tuple=True)[0]
        num_valid = len(valid_indices)
        if num_valid == 0:
            return

        count = random.randint(self.min_count, self.max_count)
        # Determine how many valid patches to erase from RE min/max count and area args
        max_erase = min(num_valid, max(1, int(num_valid * count * self.max_area)))
        min_erase = max(1, int(num_valid * count * self.min_area))
        num_erase = random.randint(min_erase, max_erase)

        # Randomly select valid patches to erase
        erase_idx = valid_indices[torch.randperm(num_valid, device=patches.device)[:num_erase]]

        if self.unique_noise_per_patch and self.erase_mode == 'pixel':
            # generate unique noise for the whole selection of patches
            fill_shape = (num_erase,) + patch_shape
        else:
            fill_shape = patch_shape

        patches[erase_idx] = self._get_values(fill_shape, dtype=dtype)

    def _erase_region(
            self,
            patches: torch.Tensor,
            patch_coord: torch.Tensor,
            patch_valid: torch.Tensor,
            patch_shape: torch.Size,
            dtype: torch.dtype = torch.float32,
    ) -> None:
        """Apply erasing by selecting rectangular regions of patches randomly.

        Closer to the original RandomErasing implementation. Erases
        spatially contiguous rectangular regions of patches (aligned with patches).

        Args:
            patches: Tensor of patches to modify in-place.
            patch_coord: Tensor of patch coordinates.
            patch_valid: Tensor indicating which patches are valid.
            patch_shape: Shape of individual patches.
            dtype: Data type for generated values.
        """
        if random.random() > self.erase_prob:
            return

        # Determine grid dimensions from coordinates
        valid_coord = patch_coord[patch_valid]
        if len(valid_coord) == 0:
            return  # No valid patches
        max_y = valid_coord[:, 0].max().item() + 1
        max_x = valid_coord[:, 1].max().item() + 1
        grid_h, grid_w = max_y, max_x
        total_area = grid_h * grid_w
        ys, xs = patch_coord[:, 0], patch_coord[:, 1]

        count = random.randint(self.min_count, self.max_count)
        for _ in range(count):
            # Try to select a valid region to erase (multiple attempts)
            for attempt in range(10):
                # Sample random area and aspect ratio
                target_area = random.uniform(self.min_area, self.max_area) * total_area
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))

                # Calculate region height and width
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if h > grid_h or w > grid_w:
                    continue  # try again

                # Calculate region patch bounds
                top = random.randint(0, grid_h - h)
                left = random.randint(0, grid_w - w)
                bottom, right = top + h, left + w

                # Region test
                region_mask = (
                        (ys >= top) & (ys < bottom) &
                        (xs >= left) & (xs < right) &
                        patch_valid
                )
                num_selected = int(region_mask.sum().item())
                if not num_selected:
                    continue  # no patch actually falls inside – try again

                if self.unique_noise_per_patch and self.erase_mode == 'pixel':
                    # generate unique noise for the whole region
                    fill_shape = (num_selected,) + patch_shape
                else:
                    fill_shape = patch_shape

                patches[region_mask] = self._get_values(fill_shape, dtype=dtype)
                # Successfully applied erasing, exit the loop
                break

    def __call__(
            self,
            patches: torch.Tensor,
            patch_coord: torch.Tensor,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply random patch erasing.

        Args:
            patches: Tensor of shape [B, N, P*P, C] or [B, N, Ph, Pw, C].
            patch_coord: Tensor of shape [B, N, 2] with (y, x) coordinates.
            patch_valid: Boolean tensor of shape [B, N] indicating which patches are valid.

        Returns:
            Erased patches tensor of same shape as input.
        """
        if patches.ndim == 4:
            batch_size, num_patches, patch_dim, channels = patches.shape
        elif patches.ndim == 5:
            batch_size, num_patches, patch_h, patch_w, channels = patches.shape
        else:
            assert False
        patch_shape = patches.shape[2:]
        # patch_shape ==> shape of patches to fill (h, w, c) or (h * w, c)

        # Create default valid mask if not provided
        if patch_valid is None:
            patch_valid = torch.ones((batch_size, num_patches), dtype=torch.bool, device=patches.device)

        # Skip the first part of the batch if num_splits is set
        batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0

        # Apply erasing to each batch element
        for i in range(batch_start, batch_size):
            if self.patch_drop_prob:
                assert False, "WIP, not completed"
                self._drop_patches(
                    patches[i],
                    patch_coord[i],
                    patch_valid[i],
                )
            elif self.spatial_mode == 'patch':
                # FIXME we could vectorize patch mode across batch, worth the effort?
                self._erase_patches(
                    patches[i],
                    patch_coord[i],
                    patch_valid[i],
                    patch_shape,
                    patches.dtype
                )
            elif self.spatial_mode == 'region':
                self._erase_region(
                    patches[i],
                    patch_coord[i],
                    patch_valid[i],
                    patch_shape,
                    patches.dtype
                )
            else:
                assert False

        return patches

    def __repr__(self) -> str:
        """Return string representation of PatchRandomErasing.

        Returns:
            String representation of the object.
        """
        fs = self.__class__.__name__ + f'(p={self.erase_prob}, mode={self.erase_mode}'
        fs += f', spatial={self.spatial_mode}, area=({self.min_area}, {self.max_area}))'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs