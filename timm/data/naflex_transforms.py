""" NaFlex (NaViT + FlexiViT) Transforms and Collation

Implements PyTorch versions of the transforms described in the NaViT and FlexiViT papers:
- NaViT: https://arxiv.org/abs/2307.14995
- FlexiViT: https://arxiv.org/abs/2212.08013

Enables variable resolution/aspect ratio image handling with efficient patching.

Hacked together by / Copyright 2025, Ross Wightman, Hugging Face
"""

import math
import random
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from .transforms import str_to_interp_mode, crop_or_pad, center_crop_or_pad


def get_image_size_for_seq(
        image_hw: Tuple[int, int],
        patch_size: Union[int, Tuple[int, int]] = 16,
        max_seq_len: int = 1024,
        divisible_by_patch: bool = True,
        max_ratio: Optional[float] = None,
        eps: float = 1e-5,
) -> Tuple[float, Tuple[int, int]]:
    """Determine scaling ratio and image size for sequence length constraint.

    Calculates the scaling ratio needed so that when image_hw is scaled,
    the total number of resulting patches does not exceed max_seq_len.

    Args:
        image_hw: Original image dimensions (height, width).
        patch_size: Patch dimensions. If int, patches are square.
        max_seq_len: Maximum allowed sequence length.
        divisible_by_patch: Whether resulting dimensions must be divisible by patch_size.
        max_ratio: Optional cap on scaling ratio to prevent excessive upsampling.
        eps: Convergence threshold for binary search.

    Returns:
        Tuple of (ratio, target_hw) where ratio is the scaling factor and
        target_hw is the resulting (height, width) after scaling.
    """

    # Handle patch size input, extract patch_h, patch_w
    if isinstance(patch_size, int):
        patch_h, patch_w = patch_size, patch_size
    else:
        # Assume it's a tuple/list: (patch_h, patch_w)
        if len(patch_size) != 2:
            raise ValueError("patch_size tuple must have exactly two elements (patch_h, patch_w).")
        patch_h, patch_w = patch_size

    # Safety checks
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_size dimensions must be positive.")

    def prepare_target_hw(ratio):
        """Scale image_hw by ratio and optionally round dimensions to multiples of patch_h, patch_w."""
        scaled_h = image_hw[0] * ratio
        scaled_w = image_hw[1] * ratio

        # If we need the result to be divisible by patch_size
        if divisible_by_patch:
            scaled_h = patch_h * math.ceil(scaled_h / patch_h)
            scaled_w = patch_w * math.ceil(scaled_w / patch_w)

        # Ensure at least one patch in each dimension
        scaled_h = int(max(scaled_h, patch_h))
        scaled_w = int(max(scaled_w, patch_w))

        return scaled_h, scaled_w

    def is_feasible(ratio):
        """Check if scaling by 'ratio' keeps patch count within max_seq_len."""
        t_h, t_w = prepare_target_hw(ratio)

        # Each dimension is already a multiple of patch_h, patch_w if divisible_by_patch=True.
        # Use integer division to count patches.
        num_patches_h = t_h // patch_h
        num_patches_w = t_w // patch_w
        seq_len = num_patches_h * num_patches_w

        return seq_len <= max_seq_len

    # Binary search boundaries
    lb = eps / 10.0
    rb = 100.0

    # Standard binary search loop
    while (rb - lb) >= eps:
        mid = (lb + rb) / 2.0
        if is_feasible(mid):
            lb = mid
        else:
            rb = mid

    # The final ratio from the binary search
    ratio = lb

    # If max_ratio is provided, clamp it to prevent upsampling beyond that threshold
    if max_ratio is not None:
        ratio = min(ratio, max_ratio)

    # Final checks
    if ratio <= eps:
        raise ValueError("Binary search failed - image might be too large?")
    if ratio >= 100.0:
        raise ValueError("Binary search failed - image might be too small?")

    # Prepare the final target dimensions with the possibly clamped ratio
    target_hw = prepare_target_hw(ratio)
    return ratio, target_hw


_RANDOM_INTERPOLATION = (str_to_interp_mode('bilinear'), str_to_interp_mode('bicubic'))


class ResizeToSequence(torch.nn.Module):
    """Resize image to fit within a maximum sequence length constraint when patchified.

    This maintains aspect ratio while ensuring the resulting image, when divided into patches,
    will not exceed the specified maximum sequence length.
    """
    def __init__(
            self,
            patch_size: int,
            max_seq_len: int = 1024,
            divisible_by_patch: bool = True,
            max_ratio: Optional[float] = None,
            interpolation: Union[str, InterpolationMode, Tuple[InterpolationMode, ...]] = 'bicubic',
        ) -> None:
        """Initialize ResizeToSequence transform.

        Args:
            patch_size: Size of patches.
            max_seq_len: Maximum sequence length constraint.
            divisible_by_patch: Whether dimensions must be divisible by patch_size.
            max_ratio: Optional cap on scaling ratio.
            interpolation: Interpolation method or methods.
        """
        super().__init__()
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.divisible_by_patch = divisible_by_patch
        self.max_ratio = max_ratio
        if isinstance(interpolation, str):
            if interpolation == 'random':
                self.interpolation = _RANDOM_INTERPOLATION
            else:
                self.interpolation = str_to_interp_mode(interpolation)
        else:
            self.interpolation = interpolation


    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Resize image to maintain aspect ratio and fit sequence constraint.

        Args:
            img: Input image tensor.

        Returns:
            Resized image tensor.
        """
        _, h, w = transforms.functional.get_dimensions(img)

        _, target_hw = get_image_size_for_seq(
            (h, w),
            self.patch_size,
            self.max_seq_len,
            divisible_by_patch=self.divisible_by_patch,
            max_ratio=self.max_ratio,
        )

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation

        resized_img = transforms.functional.resize(img, target_hw, interpolation=interpolation, antialias=True)

        return resized_img


class ResizeKeepRatioToSequence(torch.nn.Module):
    """
    Resize and Keep Aspect Ratio, adapted to fit sequence length constraints.
    """

    def __init__(
            self,
            patch_size=16,
            max_sequence_len=1024,
            divisible_by_patch=True,
            longest=0.,
            interpolation='bilinear',
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_scale_area=False,
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11),
            max_ratio=None,
    ):
        """
        Args:
            patch_size: Size of patches (int or tuple of (patch_h, patch_w))
            max_sequence_len: Maximum allowed sequence length for the resulting image
            divisible_by_patch: If True, ensure dimensions are divisible by patch_size
            longest: Float between 0-1 where 0=shortest side, 1=longest side determines scale
            interpolation: Interpolation method for resizing
            random_scale_prob: Probability of applying random scaling
            random_scale_range: Range for random scaling factor (min, max)
            random_scale_area: If True, scale factors affect area (√ factor)
            random_aspect_prob: Probability of applying random aspect ratio jittering
            random_aspect_range: Range for random aspect ratio (min, max)
            max_ratio: Maximum allowed scaling ratio
        """
        super().__init__()
        self.patch_size = patch_size
        self.max_sequence_len = max_sequence_len
        self.divisible_by_patch = divisible_by_patch
        self.longest = float(longest)

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)

        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_scale_area = random_scale_area
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range
        self.max_ratio = max_ratio

    @staticmethod
    def get_params(
            img,
            patch_size,
            max_sequence_len,
            divisible_by_patch,
            longest,
            random_scale_prob=0.,
            random_scale_range=(1.0, 1.33),
            random_scale_area=False,
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11),
            max_ratio=None,
    ):
        """Get parameters for resizing."""
        # Get image dimensions
        img_h, img_w = F.get_dimensions(img)[1:]

        # Step 1: Get the maximum allowed dimensions from sequence length constraint
        _, target_hw = get_image_size_for_seq(
            (img_h, img_w),
            patch_size,
            max_sequence_len,
            divisible_by_patch,
            max_ratio,
        )
        target_h, target_w = target_hw

        # Calculate ratio based on sequence constraint
        ratio_h = target_h / img_h
        ratio_w = target_w / img_w
        # Apply longest blending
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)

        # Apply random scaling
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            if random_scale_area:
                # Make ratio factor equivalent to area change
                ratio_factor = 1. / math.sqrt(ratio_factor)
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)

        # Apply random aspect
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            log_aspect = (math.log(random_aspect_range[0]), math.log(random_aspect_range[1]))
            aspect_factor = math.exp(random.uniform(*log_aspect))
            aspect_factor = math.sqrt(aspect_factor)
            # Apply aspect ratio jittering
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)

        # Calculate final dimensions
        size = [round(dim * ratio * f) for dim, f in zip((img_h, img_w), ratio_factor)]

        # Ensure dimensions satisfy sequence constraint and are divisible by patch size
        if isinstance(patch_size, int):
            ph, pw = patch_size, patch_size
        else:
            ph, pw = patch_size

        # Ensure dimensions are at least one patch
        size[0] = max(size[0], ph)
        size[1] = max(size[1], pw)

        # Make divisible by patch size if needed
        if divisible_by_patch:
            size[0] = ph * math.ceil(size[0] / ph)
            size[1] = pw * math.ceil(size[1] / pw)

        # Verify we haven't exceeded sequence length
        num_patches_h = size[0] // ph
        num_patches_w = size[1] // pw
        seq_len = num_patches_h * num_patches_w

        if seq_len > max_sequence_len:
            # Scale back down to fit sequence constraint
            scale_back = math.sqrt(max_sequence_len / seq_len)
            size[0] = int(size[0] * scale_back)
            size[1] = int(size[1] * scale_back)

            # Ensure divisible by patch size after scaling back
            if divisible_by_patch:
                size[0] = ph * math.ceil(size[0] / ph)
                size[1] = pw * math.ceil(size[1] / pw)

        return size

    def forward(self, img):
        """
        Resize the image with aspect ratio preservation and sequence length constraints.
        """
        size = self.get_params(
            img,
            self.patch_size,
            self.max_sequence_len,
            self.divisible_by_patch,
            self.longest,
            self.random_scale_prob,
            self.random_scale_range,
            self.random_scale_area,
            self.random_aspect_prob,
            self.random_aspect_range,
            self.max_ratio,
        )

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation

        return F.resize(img, size, interpolation)

    def __repr__(self):
        interpolate_str = "random" if isinstance(self.interpolation, (tuple, list)) else str(self.interpolation)
        return (f"{self.__class__.__name__}(patch_size={self.patch_size}, "
                f"max_sequence_len={self.max_sequence_len}, "
                f"longest={self.longest:.3f}, "
                f"random_scale_prob={self.random_scale_prob:.3f}, "
                f"random_aspect_prob={self.random_aspect_prob:.3f})")


class CenterCropToSequence(torch.nn.Module):
    """Center crop the image such that the resulting patch sequence length meets constraints."""
    def __init__(
            self,
            patch_size: int,
            max_seq_len: int,
            divisible_by_patch: bool = True,
            fill: Union[int, Tuple[int, int, int]] = 0,
            padding_mode: str = 'constant'
        ):
        super().__init__()
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.divisible_by_patch = divisible_by_patch
        self.fill = fill
        self.padding_mode = padding_mode


    def forward(self, img):
        """Center crop the image to maintain aspect ratio and fit sequence constraint."""
        _, h, w = transforms.functional.get_dimensions(img)
        _, target_hw = get_image_size_for_seq(
            (h, w),
            self.patch_size,
            self.max_seq_len,
            self.divisible_by_patch
        )

        # Use center crop
        return center_crop_or_pad(img, target_hw, fill=self.fill, padding_mode=self.padding_mode)


class RandomCropToSequence(torch.nn.Module):
    """Randomly crop and/or pad the image to fit sequence length constraints.

    This maintains aspect ratio while ensuring the resulting image, when divided into patches,
    will not exceed the specified maximum sequence length. Similar to CentralCropToSequence
    but with randomized positioning.
    """

    def __init__(
            self,
            patch_size: int,
            max_sequence_len: int,
            divisible_by_patch: bool = True,
            fill: Union[int, Tuple[int, int, int]] = 0,
            padding_mode: str = 'constant'
    ):
        """
        Args:
            patch_size: Size of patches (int or tuple of (patch_h, patch_w))
            max_sequence_len: Maximum allowed sequence length for the resulting image
            divisible_by_patch: If True, resulting image dimensions will be multiples of patch_size
            fill: Fill value for padding
            padding_mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
        """
        super().__init__()
        self.patch_size = patch_size
        self.max_sequence_len = max_sequence_len
        self.divisible_by_patch = divisible_by_patch
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, target_size):
        """Get random position for crop/pad."""
        _, image_height, image_width = transforms.functional.get_dimensions(img)
        delta_height = image_height - target_size[0]
        delta_width = image_width - target_size[1]

        # Handle both positive (crop) and negative (pad) deltas
        if delta_height == 0:
            top = 0
        else:
            top = int(math.copysign(random.randint(0, abs(delta_height)), delta_height))

        if delta_width == 0:
            left = 0
        else:
            left = int(math.copysign(random.randint(0, abs(delta_width)), delta_width))

        return top, left

    def forward(self, img):
        """Randomly crop or pad the image to maintain aspect ratio and fit sequence constraint."""
        # Get current dimensions
        _, img_h, img_w = transforms.functional.get_dimensions(img)

        # Calculate target dimensions that satisfy sequence length
        # We use max_ratio=1.0 to prevent upscaling - we only want to crop or maintain current size
        _, target_hw = get_image_size_for_seq(
            (img_h, img_w),
            self.patch_size,
            self.max_sequence_len,
            self.divisible_by_patch,
            max_ratio=1.0  # Prevent upscaling
        )

        # Get random position for crop/pad
        top, left = self.get_params(img, target_hw)

        # Apply crop or pad
        return crop_or_pad(
            img,
            top=top,
            left=left,
            height=target_hw[0],
            width=target_hw[1],
            fill=self.fill,
            padding_mode=self.padding_mode,
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(patch_size={self.patch_size}, "
                f"max_sequence_len={self.max_sequence_len}, "
                f"divisible_by_patch={self.divisible_by_patch})")


def _validate_range(value, name, length=2):
    # Validate type and length
    if not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{name} should be a sequence of length {length}.")

    # Validate order
    if value[0] > value[1]:
        warnings.warn(f"{name.capitalize()} range reversed. Swapping.")
        return value[1], value[0]

    return value


class RandomResizedCropToSequence(torch.nn.Module):
    """
    Randomly crop the input image to a subregion with varying area and aspect ratio
    (relative to the original), then resize that crop to a target size. The target size
    is determined such that patchifying the resized image (with `patch_size`)
    does not exceed `max_seq_len` patches, while maintaining the aspect ratio of the crop.

    This combines aspects of torchvision's RandomResizedCrop with sequence length constraints.

    Args:
        patch_size (int or tuple[int, int]):
            Patch dimensions (patch_h, patch_w) for sequence length calculation.
        max_seq_len (int):
            Maximum number of patches allowed in the final image.
        scale (tuple[float, float]):
            Range (min, max) of area fraction of the original image to crop.
        ratio (tuple[float, float]):
            Range (min, max) of aspect ratio *multipliers* for the crop, relative
            to the original image's aspect ratio. E.g., (0.75, 1.333) means the
            crop's aspect ratio will be sampled between 0.75*orig_ar and 1.333*orig_ar.
            Uses log-uniform sampling.
        interpolation (str or InterpolationMode):
            Interpolation mode for resizing. Can be 'bilinear', 'bicubic', 'nearest',
            or 'random' (chooses between bilinear and bicubic).
            Defaults to 'bicubic'.
        divisible_by_patch (bool):
            If True, the final image height and width will be multiples of the
            respective patch dimensions. Defaults to True.
        max_ratio (float, optional):
            An optional upper limit on the scaling ratio applied during resizing.
            Prevents excessive upsampling of the initial crop. `max_ratio=1.0`
            prevents any upsampling beyond the cropped size. Defaults to None (no limit).
        final_scale_range (tuple[float, float], optional):
            If provided, applies an *additional* random scaling factor to the
            final target size. The factor is sampled uniformly from this range,
            and multiplied by the size determined by `get_image_size_for_seq`.
            E.g., (0.8, 1.0) means the final size will be between 80% and 100%
            of the maximum feasible size. Defaults to None (use maximum feasible size).
        attempts (int):
            Number of attempts to sample a valid crop geometry before falling back
            to a center crop strategy. Defaults to 10.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        max_seq_len: int = 1024,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (.8, 1.25),
        interpolation: Union[str, InterpolationMode] = 'bicubic',
        divisible_by_patch: bool = True,
        max_ratio: Optional[float] = None,
        final_scale_range: Optional[Tuple[float, float]] = None,
        attempts: int = 10,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_h, self.patch_w = patch_size, patch_size
        else:
            # Assume it's a tuple/list: (patch_h, patch_w)
            if len(patch_size) != 2:
                raise ValueError("patch_size tuple must have exactly two elements (patch_h, patch_w).")
            self.patch_h, self.patch_w = patch_size
        self.max_seq_len = max_seq_len
        self.scale = scale
        self.ratio = ratio
        self.divisible_by_patch = divisible_by_patch
        self.max_ratio = max_ratio
        self.final_scale_range = final_scale_range
        self.attempts = attempts
        if isinstance(interpolation, str):
            if interpolation == 'random':
                self.interpolation = _RANDOM_INTERPOLATION
            else:
                self.interpolation = str_to_interp_mode(interpolation)
        else:
            self.interpolation = interpolation

        # Validate scale and ratio
        self.scale = _validate_range(self.scale, "scale")
        self.ratio = _validate_range(self.ratio, "ratio")

        # Validate final_scale_range if provided
        if self.final_scale_range is not None:
            self.final_scale_range = _validate_range(self.final_scale_range, "final_scale_range")

            # Additional validation for final_scale_range values
            if not (0.0 <= self.final_scale_range[0] <= self.final_scale_range[1] <= 1.0):
                warnings.warn("final_scale_range values should ideally be between 0.0 and 1.0.")

    @staticmethod
    def get_params(
            img: torch.Tensor,
            scale: Tuple[float, float],
            ratio: Tuple[float, float],
            crop_attempts: int = 10,
            patch_h: int = 16,
            patch_w: int = 16,
            max_seq_len: int = 1024,
            divisible_by_patch: bool = True,
            max_ratio: Optional[float] = None,
            final_scale_range: Optional[Tuple[float, float]] = None,
            interpolation: Union[List[InterpolationMode], InterpolationMode] = _RANDOM_INTERPOLATION,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int], InterpolationMode]:
        """ Get parameters for a random sized crop relative to image aspect ratio.
        """
        _, height, width = F.get_dimensions(img)
        if height <= 0 or width <= 0:
             raise ValueError(f"Input image must have positive dimensions, got H={height}, W={width}")

        area = height * width
        orig_aspect = width / height
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

        for _ in range(crop_attempts):
            target_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio_factor = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
            aspect_ratio = orig_aspect * aspect_ratio_factor

            # Calculate target dimensions for the crop
            # target_area = crop_w * crop_h, aspect_ratio = crop_w / crop_h
            # => crop_h = sqrt(target_area / aspect_ratio)
            # => crop_w = sqrt(target_area * aspect_ratio)
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))

            if 0 < crop_w <= width and 0 < crop_h <= height:
                top = random.randint(0, height - crop_h)
                left = random.randint(0, width - crop_w)
                break
        else:
            # Fallback strategy, use center crop trying to respect ratio range
            min_aspect_ratio = orig_aspect * ratio[0]
            max_aspect_ratio = orig_aspect * ratio[1]

            if orig_aspect < min_aspect_ratio:
                # Original is narrower than target min, clamp width
                crop_w = width
                crop_h = min(int(round(crop_w / min_aspect_ratio)), height)
            elif orig_aspect > max_aspect_ratio:
                # Original is wider than target max, clamp height
                crop_h = height
                crop_w = min(int(round(crop_h * max_aspect_ratio)), width)
            else:
                # Aspect ratio is within range, take the largest possible crop (full image)
                crop_w = width
                crop_h = height

            # Ensure valid dimensions after fallback calculation
            crop_h = max(1, crop_h)
            crop_w = max(1, crop_w)

            top = (height - crop_h) // 2
            left = (width - crop_w) // 2

        # Determine max feasible size for scaling of the *cropped* region
        feasible_ratio, feasible_size = get_image_size_for_seq(
            (crop_h, crop_w),
            patch_size=(patch_h, patch_w), # Pass as tuple
            max_seq_len=max_seq_len,
            divisible_by_patch=divisible_by_patch,
            max_ratio=max_ratio,
        )

        # Optionally apply final scale randomization
        final_size = feasible_size
        if final_scale_range is not None:
            min_sc, max_sc = final_scale_range
            scale_factor = random.uniform(min_sc, max_sc)
            scale_factor = min(max(scale_factor, 0.0), 1.0) # Clamp factor just in case

            # Calculate raw scaled size
            # Note: feasible_ratio already accounts for max_ratio clamp if any
            raw_h = crop_h * feasible_ratio * scale_factor
            raw_w = crop_w * feasible_ratio * scale_factor

            # Re-apply divisibility constraint if needed
            if divisible_by_patch:
                # Use ceil to avoid going under minimum patch size
                target_h = patch_h * math.ceil(raw_h / patch_h)
                target_w = patch_w * math.ceil(raw_w / patch_w)
            else:
                target_h = int(round(raw_h))
                target_w = int(round(raw_w))

            # Ensure final size is at least one patch dimension
            target_h = max(target_h, patch_h)
            target_w = max(target_w, patch_w)
            final_size = (target_h, target_w)

             # Final check: Ensure this randomized size still fits max_seq_len
             # (It should, as we scaled down, but rounding might theoretically push it over)
            num_patches_h = final_size[0] // patch_h
            num_patches_w = final_size[1] // patch_w
            if (num_patches_h * num_patches_w) > max_seq_len:
                 # If it exceeds, revert to the original feasible_size (safest)
                 final_size = feasible_size
                 warnings.warn(f"Final scale randomization ({scale_factor:.2f}) resulted in size {final_size} exceeding max_seq_len={max_seq_len} after rounding. Reverting to feasible size {feasible_size}.")

        # Select interpolation mode
        if isinstance(interpolation, (tuple, list)):
            interpolation = random.choice(interpolation)
        else:
            interpolation = interpolation

        return (top, left, crop_h, crop_w), final_size, interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Sample crop, resize, and interpolation parameters
        crop_params, final_size, interpolation = self.get_params(
            img,
            scale=self.scale,
            ratio=self.ratio,
            crop_attempts=self.attempts,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            divisible_by_patch=self.divisible_by_patch,
            max_seq_len=self.max_seq_len,
            final_scale_range=self.final_scale_range,
            interpolation=self.interpolation,
        )
        top, left, crop_h, crop_w = crop_params

        output = F.resized_crop(
            img,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=final_size,
            interpolation=interpolation,
            antialias=True,
        )

        return output

    def __repr__(self) -> str:
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ', '.join(str(m).split('.')[-1] for m in self.interpolation)
        else:
            interpolate_str = str(self.interpolation)
        format_string = self.__class__.__name__ + '('
        format_string += f"patch_size=({self.patch_h}, {self.patch_w})"
        format_string += f", max_seq_len={self.max_seq_len}"
        format_string += f", scale={self.scale}"
        format_string += f", ratio={self.ratio}"
        format_string += f", interpolation=[{interpolate_str}]"
        format_string += f", divisible_by_patch={self.divisible_by_patch}"
        format_string += f", max_ratio={self.max_ratio}"
        format_string += f", final_scale_range={self.final_scale_range}"
        format_string += f", attempts={self.attempts}"
        format_string += ')'
        return format_string


def patchify_image(
        img: torch.Tensor,
        patch_size: Tuple[int, int],
        pad: bool = True,
        include_info: bool = True,
        flatten_patches: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    c, h, w = img.shape
    ph, pw = patch_size

    # Ensure the image is divisible by patch size
    if pad and (h % ph != 0 or w % pw != 0):
        pad_h = (ph - h % ph) % ph  # amount to add on bottom
        pad_w = (pw - w % pw) % pw  # amount to add on right
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
        c, h, w = img.shape

    # Calculate number of patches in each dimension
    nh, nw = h // ph, w // pw
    # Reshape image to patches
    patches = img.view(c, nh, ph, nw, pw).permute(1, 3, 2, 4, 0)
    # [nh, nw, ph, pw, c] -> [nh * nw, ph * pw * c] or [nh * nw, ph, pw, c]
    patches = patches.reshape(-1, ph * pw * c) if flatten_patches else patches.reshape(-1, ph, pw, c)

    if include_info:
        # Create coordinate indices
        y_idx, x_idx = torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing='ij')
        # Stack into a single coords tensor [N, 2] with (y, x) order
        coord = torch.stack([y_idx.reshape(-1), x_idx.reshape(-1)], dim=1)
        # Create type indicators (all 1s for regular patches)
        valid = torch.ones(nh * nw, dtype=torch.bool)
        return patches, coord, valid

    return patches


class Patchify(torch.nn.Module):
    """Transform an image into patches with corresponding coordinates and type indicators."""

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]],
            flatten_patches: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.flatten_patches = flatten_patches

    def forward(self, img):
        """
        Args:
            img: A PIL Image or tensor of shape [C, H, W]

        Returns:
            A dictionary containing:
                - patches: Tensor of shape [N, P*P*C] if flatten_patches=True,
                          or [N, Ph, Pw, C] if flatten_patches=False
                - patch_coord: Tensor of shape [N, 2] with (y, x) coordinates
                - patch_valid: Valid indicator (all 1s for non-padding patches)
        """
        if isinstance(img, Image.Image):
            # Convert PIL Image to tensor [C, H, W]
            img = transforms.functional.to_tensor(img)

        patches, coord, valid = patchify_image(img, self.patch_size, flatten_patches=self.flatten_patches)

        return {
            'patches': patches,
            'patch_coord': coord,
            'patch_valid': valid,
        }
