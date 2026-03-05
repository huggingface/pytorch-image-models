"""Multi-view transforms for self-supervised learning.

Creates multiple augmented views of each image for contrastive/joint-embedding methods
like LeJEPA, DINO, SimCLR, etc.
"""
from typing import Callable, List, Optional, Union

import torch


class MultiViewTransform:
    """Apply multiple transforms to create different views of an image.

    Each transform is applied independently to the same input image,
    producing multiple augmented views for self-supervised learning.

    Args:
        transforms: List of transform callables. Each transform is applied
            to the input image to produce one view.

    Example:
        >>> from timm.data import create_transform
        >>> # Create 2 views with same augmentation settings (different random samples)
        >>> t1 = create_transform(224, is_training=True, auto_augment='rand-m9-mstd0.5')
        >>> t2 = create_transform(224, is_training=True, auto_augment='rand-m9-mstd0.5')
        >>> multi_view = MultiViewTransform([t1, t2])
        >>> views = multi_view(pil_image)  # Returns list of 2 tensors
    """

    def __init__(self, transforms: List[Callable]):
        if not transforms:
            raise ValueError("transforms list cannot be empty")
        self.transforms = transforms
        self.num_views = len(transforms)

    def __call__(self, img):
        """Apply all transforms to the image.

        Args:
            img: Input image (PIL Image or tensor depending on transform expectations)

        Returns:
            List of transformed images, one per transform
        """
        return [t(img) for t in self.transforms]

    def __repr__(self):
        return f"{self.__class__.__name__}(num_views={self.num_views})"


class MultiViewCollator:
    """Collate function for multi-view datasets.

    Stacks multi-view samples into a single tensor of shape [B, V, C, H, W]
    where B is batch size and V is number of views.

    Args:
        stack_views: If True (default), stack views into [B, V, C, H, W].
            If False, return list of [B, C, H, W] tensors.
    """

    def __init__(self, stack_views: bool = True):
        self.stack_views = stack_views

    def __call__(self, batch):
        """Collate a batch of multi-view samples.

        Args:
            batch: List of (views, target) tuples where views is a list of tensors

        Returns:
            Tuple of (images, targets) where images is [B, V, C, H, W] tensor
        """
        views_list = []
        targets = []

        for views, target in batch:
            # views is a list of V tensors, each [C, H, W]
            views_list.append(torch.stack(views))  # [V, C, H, W]
            targets.append(target)

        if self.stack_views:
            images = torch.stack(views_list)  # [B, V, C, H, W]
        else:
            # Transpose to list of V tensors, each [B, C, H, W]
            num_views = len(views_list[0])
            images = [
                torch.stack([v[i] for v in views_list])
                for i in range(num_views)
            ]

        targets = torch.tensor(targets)
        return images, targets
