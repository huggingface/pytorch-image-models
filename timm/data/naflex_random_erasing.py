import random
import math
from typing import Optional, Union, Tuple

import torch


class PatchRandomErasing:
    """
    Random erasing for patchified images in NaFlex format.

    Supports three modes:
    1. 'patch': Simple mode that erases randomly selected valid patches
    2. 'region': Erases spatial regions at patch granularity
    3. 'subregion': Most sophisticated mode that erases spatial regions at sub-patch granularity,
       partially erasing patches that are on the boundary of the erased region

    Args:
        erase_prob: Probability that the Random Erasing operation will be performed.
        patch_drop_prob: Patch dropout probability. Remove random patches instead of erasing.
        min_area: Minimum percentage of valid patches/area to erase.
        max_area: Maximum percentage of valid patches/area to erase.
        min_aspect: Minimum aspect ratio of erased area (only used in 'region'/'subregion' mode).
        max_aspect: Maximum aspect ratio of erased area (only used in 'region'/'subregion' mode).
        mode: Patch content mode, one of 'const', 'rand', or 'pixel'
            'const' - erase patch is constant color of 0 for all channels
            'rand'  - erase patch has same random (normal) value across all elements
            'pixel' - erase patch has per-element random (normal) values
        spatial_mode: Erasing strategy, one of 'patch', 'region', or 'subregion'
        patch_size: Size of each patch (required for 'subregion' mode)
        num_splits: Number of splits to apply erasing to (0 for all)
        device: Computation device
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
            patch_size: Optional[Union[int, Tuple[int, int]]] = 16,
            num_splits: int = 0,
            device: Union[str, torch.device] = 'cuda',
    ):
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

        # Patch size (needed for subregion mode)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        # Value generation mode flags
        self.erase_mode = mode.lower()
        assert self.erase_mode in ('rand', 'pixel', 'const')
        self.const_value = value

    def _get_values(
            self,
            shape: Union[Tuple[int,...], torch.Size],
            value: Optional[torch.Tensor] = None,
            dtype: torch.dtype = torch.float32,
            device: Optional[Union[str, torch.device]] = None
    ):
        """Generate values for erased patches based on the specified mode.
        Args:
            shape: Shape of patches to erase.
            value: Value to use in const (or rand) mode.
            dtype: Data type to use.
            device: Device to use.
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
    ):
        """ Patch Dropout

        Fully drops patches from datastream. Only mode that saves compute BUT requires support
        for non-contiguous patches and associated patch coordinate and valid handling.
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
    ):
        """Apply erasing by selecting individual patches randomly.

        The simplest mode, aligned on patch boundaries. Behaves similarly to speckle or 'sprinkles'
        noise augmentation at patch size.
        """
        if random.random() > self.erase_prob:
            return

        # Get indices of valid patches
        valid_indices = torch.nonzero(patch_valid, as_tuple=True)[0].tolist()
        if not valid_indices:
            # Skip if no valid patches
            return

        num_valid = len(valid_indices)
        count = random.randint(self.min_count, self.max_count)
        # Determine how many valid patches to erase from RE min/max count and area args
        max_erase = max(1, int(num_valid * count * self.max_area))
        min_erase = max(1, int(num_valid * count * self.min_area))
        num_erase = random.randint(min_erase, max_erase)

        # Randomly select valid patches to erase
        indices_to_erase = random.sample(valid_indices, min(num_erase, num_valid))

        random_value = None
        if self.erase_mode == 'rand':
            random_value = torch.empty(patch_shape[-1], dtype=dtype, device=self.device).normal_()

        for idx in indices_to_erase:
            patches[idx].copy_(self._get_values(patch_shape, dtype=dtype, value=random_value))

    def _erase_region(
            self,
            patches: torch.Tensor,
            patch_coord: torch.Tensor,
            patch_valid: torch.Tensor,
            patch_shape: torch.Size,
            dtype: torch.dtype = torch.float32,
    ):
        """Apply erasing by selecting rectangular regions of patches randomly

        Closer to the original RandomErasing implementation. Erases
        spatially contiguous rectangular regions of patches (aligned with patches).
        """
        if random.random() > self.erase_prob:
            return

        # Determine grid dimensions from coordinates
        if patch_valid is not None:
            valid_coord = patch_coord[patch_valid]
            if len(valid_coord) == 0:
                return  # No valid patches
            max_y = valid_coord[:, 0].max().item() + 1
            max_x = valid_coord[:, 1].max().item() + 1
        else:
            max_y = patch_coord[:, 0].max().item() + 1
            max_x = patch_coord[:, 1].max().item() + 1

        grid_h, grid_w = max_y, max_x

        # Calculate total area
        total_area = grid_h * grid_w

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

                # Ensure region fits within grid
                if w <= grid_w and h <= grid_h:
                    # Select random top-left corner
                    top = random.randint(0, grid_h - h)
                    left = random.randint(0, grid_w - w)

                    # Define region bounds
                    bottom = top + h
                    right = left + w

                    # Create a single random value for all affected patches if using 'rand' mode
                    if self.erase_mode == 'rand':
                        random_value = torch.empty(patch_shape[-1], dtype=dtype, device=self.device).normal_()
                    else:
                        random_value = None

                    # Find and erase all patches that fall within the region
                    for i in range(len(patches)):
                        if patch_valid is None or patch_valid[i]:
                            y, x = patch_coord[i]
                            if top <= y < bottom and left <= x < right:
                                patches[i] = self._get_values(patch_shape, dtype=dtype, value=random_value)

                    # Successfully applied erasing, exit the loop
                    break

    def _erase_subregion(
            self,
            patches: torch.Tensor,
            patch_coord: torch.Tensor,
            patch_valid: torch.Tensor,
            patch_shape: torch.Size,
            patch_size: Tuple[int, int],
            dtype: torch.dtype = torch.float32,
    ):
        """Apply erasing by selecting rectangular regions ignoring patch boundaries.

        Matches or original RandomErasing implementation. Erases spatially contiguous rectangular
        regions that are not aligned to patches (erase regions boundaries cut within patches).

        FIXME complexity probably not worth it, may remove.
        """
        if random.random() > self.erase_prob:
            return

        # Get patch dimensions
        patch_h, patch_w = patch_size
        channels = patch_shape[-1]

        # Determine grid dimensions in patch coordinates
        if patch_valid is not None:
            valid_coord = patch_coord[patch_valid]
            if len(valid_coord) == 0:
                return  # No valid patches
            max_y = valid_coord[:, 0].max().item() + 1
            max_x = valid_coord[:, 1].max().item() + 1
        else:
            max_y = patch_coord[:, 0].max().item() + 1
            max_x = patch_coord[:, 1].max().item() + 1

        grid_h, grid_w = max_y, max_x

        # Calculate total area in pixel space
        total_area = (grid_h * patch_h) * (grid_w * patch_w)

        count = random.randint(self.min_count, self.max_count)
        for _ in range(count):
            # Try to select a valid region to erase (multiple attempts)
            for attempt in range(10):
                # Sample random area and aspect ratio
                target_area = random.uniform(self.min_area, self.max_area) * total_area
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))

                # Calculate region height and width in pixel space
                pixel_h = int(round(math.sqrt(target_area * aspect_ratio)))
                pixel_w = int(round(math.sqrt(target_area / aspect_ratio)))

                # Ensure region fits within total pixel grid
                if pixel_w <= grid_w * patch_w and pixel_h <= grid_h * patch_h:
                    # Select random top-left corner in pixel space
                    pixel_top = random.randint(0, grid_h * patch_h - pixel_h)
                    pixel_left = random.randint(0, grid_w * patch_w - pixel_w)

                    # Define region bounds in pixel space
                    pixel_bottom = pixel_top + pixel_h
                    pixel_right = pixel_left + pixel_w

                    # Create a single random value for the entire region if using 'rand' mode
                    rand_value = None
                    if self.erase_mode == 'rand':
                        rand_value = torch.empty(patch_shape[-1], dtype=dtype, device=self.device).normal_()

                    # For each valid patch, determine if and how it overlaps with the erase region
                    for i in range(len(patches)):
                        if patch_valid is None or patch_valid[i]:
                            # Convert patch coordinates to pixel space (top-left corner)
                            y, x = patch_coord[i]
                            patch_pixel_top = y * patch_h
                            patch_pixel_left = x * patch_w
                            patch_pixel_bottom = patch_pixel_top + patch_h
                            patch_pixel_right = patch_pixel_left + patch_w

                            # Check if this patch overlaps with the erase region
                            if not (patch_pixel_right <= pixel_left or patch_pixel_left >= pixel_right or
                                    patch_pixel_bottom <= pixel_top or patch_pixel_top >= pixel_bottom):

                                # Calculate the overlap region in patch-local coordinates
                                local_top = max(0, pixel_top - patch_pixel_top)
                                local_left = max(0, pixel_left - patch_pixel_left)
                                local_bottom = min(patch_h, pixel_bottom - patch_pixel_top)
                                local_right = min(patch_w, pixel_right - patch_pixel_left)

                                # Reshape the patch to [patch_h, patch_w, chans]
                                patch_data = patches[i].reshape(patch_h, patch_w, channels)

                                erase_shape = (local_bottom - local_top, local_right - local_left, channels)
                                erase_value = self._get_values(erase_shape, dtype=dtype, value=rand_value)
                                patch_data[local_top:local_bottom, local_left:local_right, :] = erase_value

                                # Flatten the patch back to [patch_h*patch_w, chans]
                                if len(patch_shape) == 2:
                                    patch_data = patch_data.reshape(-1, channels)
                                patches[i] = patch_data

                    # Successfully applied erasing, exit the loop
                    break

    def __call__(
            self,
            patches: torch.Tensor,
            patch_coord: torch.Tensor,
            patch_valid: Optional[torch.Tensor] = None,
    ):
        """
        Apply random patch erasing.

        Args:
            patches: Tensor of shape [B, N, P*P, C]
            patch_coord: Tensor of shape [B, N, 2] with (y, x) coordinates
            patch_valid: Boolean tensor of shape [B, N] indicating which patches are valid
                        If None, all patches are considered valid

        Returns:
            Erased patches tensor of same shape
        """
        if patches.ndim == 4:
            batch_size, num_patches, patch_dim, channels = patches.shape
            if self.patch_size is not None:
                patch_size = self.patch_size
            else:
                patch_size = None
        elif patches.ndim == 5:
            batch_size, num_patches, patch_h, patch_w, channels = patches.shape
            patch_size = (patch_h, patch_w)
        else:
            assert False
        patch_shape = patches.shape[2:]
        # patch_shape ==> shape of patches to fill (h, w, c) or (h * w, c)
        # patch_size ==> patch h, w (if available, must be avail for subregion mode)

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
            elif self.spatial_mode == 'subregion':
                self._erase_subregion(
                    patches[i],
                    patch_coord[i],
                    patch_valid[i],
                    patch_shape,
                    patch_size,
                    patches.dtype
                )

        return patches

    def __repr__(self):
        fs = self.__class__.__name__ + f'(p={self.erase_prob}, mode={self.erase_mode}'
        fs += f', spatial={self.spatial_mode}, area=({self.min_area}, {self.max_area}))'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs