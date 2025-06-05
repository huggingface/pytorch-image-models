"""Variable‑size Mixup / CutMix utilities for NaFlex data loaders.

This module provides:

* `mix_batch_variable_size` – pixel‑level Mixup/CutMix that operates on a
  list of images whose spatial sizes differ, mixing only their central overlap
  so no resizing is required.
* `pairwise_mixup_target` – builds soft‑label targets that exactly match the
  per‑sample pixel provenance produced by the mixer.
* `NaFlexMixup` – a callable functor that wraps the two helpers and stores
  all augmentation hyper‑parameters in one place, making it easy to plug into
  different dataset wrappers.

Hacked together by / Copyright 2025, Ross Wightman, Hugging Face
"""
import math
import random
from typing import Dict, List, Tuple, Union

import torch


def mix_batch_variable_size(
        imgs: List[torch.Tensor],
        *,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        switch_prob: float = 0.5,
        local_shuffle: int = 4,
) -> Tuple[List[torch.Tensor], List[float], Dict[int, int]]:
    """Apply Mixup or CutMix on a batch of variable-sized images.

    Sorts images by aspect ratio and pairs neighboring samples. Only the mutual
    central overlap region of each pair is mixed.

    Args:
        imgs: List of transformed images shaped (C, H, W).
        mixup_alpha: Beta distribution alpha for Mixup. Set to 0 to disable.
        cutmix_alpha: Beta distribution alpha for CutMix. Set to 0 to disable.
        switch_prob: Probability of using CutMix when both modes are enabled.
        local_shuffle: Size of local windows for shuffling after aspect sorting.

    Returns:
        Tuple of (mixed_imgs, lam_list, pair_to) where:
            - mixed_imgs: List of mixed images
            - lam_list: Per-sample lambda values representing mixing degree
            - pair_to: Mapping i -> j of which sample was mixed with which
    """
    if len(imgs) < 2:
        raise ValueError("Need at least two images to perform Mixup/CutMix.")

    # Decide augmentation mode and raw λ
    if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
        use_cutmix = torch.rand(()).item() < switch_prob
        alpha = cutmix_alpha if use_cutmix else mixup_alpha
    elif mixup_alpha > 0.0:
        use_cutmix = False
        alpha = mixup_alpha
    elif cutmix_alpha > 0.0:
        use_cutmix = True
        alpha = cutmix_alpha
    else:
        raise ValueError("Both mixup_alpha and cutmix_alpha are zero – nothing to do.")

    lam_raw = torch.distributions.Beta(alpha, alpha).sample().item()
    lam_raw = max(0.0, min(1.0, lam_raw))  # numerical safety

    # Pair images by nearest aspect ratio
    order = sorted(range(len(imgs)), key=lambda i: imgs[i].shape[2] / imgs[i].shape[1])
    if local_shuffle > 1:
        for start in range(0, len(order), local_shuffle):
            random.shuffle(order[start:start + local_shuffle])

    pair_to: Dict[int, int] = {}
    for a, b in zip(order[::2], order[1::2]):
        pair_to[a] = b
        pair_to[b] = a

    odd_one = order[-1] if len(imgs) % 2 else None

    mixed_imgs: List[torch.Tensor] = [None] * len(imgs)
    lam_list: List[float] = [1.0] * len(imgs)

    for i in range(len(imgs)):
        if i == odd_one:
            mixed_imgs[i] = imgs[i]
            continue

        j = pair_to[i]
        xi, xj = imgs[i], imgs[j]
        _, hi, wi = xi.shape
        _, hj, wj = xj.shape
        dest_area = hi * wi

        # Central overlap common to both images
        oh, ow = min(hi, hj), min(wi, wj)
        overlap_area = oh * ow
        top_i, left_i = (hi - oh) // 2, (wi - ow) // 2
        top_j, left_j = (hj - oh) // 2, (wj - ow) // 2

        xi = xi.clone()
        if use_cutmix:
            # CutMix: random rectangle inside the overlap
            cut_ratio = math.sqrt(1.0 - lam_raw)
            ch, cw = int(oh * cut_ratio), int(ow * cut_ratio)
            cut_area = ch * cw
            y_off = random.randint(0, oh - ch)
            x_off = random.randint(0, ow - cw)

            yl_i, xl_i = top_i + y_off, left_i + x_off
            yl_j, xl_j = top_j + y_off, left_j + x_off
            xi[:, yl_i: yl_i + ch, xl_i: xl_i + cw] = xj[:, yl_j: yl_j + ch, xl_j: xl_j + cw]
            mixed_imgs[i] = xi

            corrected_lam = 1.0 - cut_area / float(dest_area)
            lam_list[i] = corrected_lam
        else:
            # Mixup: blend the entire overlap region
            patch_i = xi[:, top_i:top_i + oh, left_i:left_i + ow]
            patch_j = xj[:, top_j:top_j + oh, left_j:left_j + ow]

            blended = patch_i.mul(lam_raw).add_(patch_j, alpha=1.0 - lam_raw)
            xi[:, top_i:top_i + oh, left_i:left_i + ow] = blended
            mixed_imgs[i] = xi

            corrected_lam = (dest_area - overlap_area) / dest_area + lam_raw * overlap_area / dest_area
            lam_list[i] = corrected_lam

    return mixed_imgs, lam_list, pair_to


def smoothed_sparse_target(
        targets: torch.Tensor,
        *,
        num_classes: int,
        smoothing: float = 0.0,
) -> torch.Tensor:
    off_val = smoothing / num_classes
    on_val = 1.0 - smoothing + off_val

    y_onehot = torch.full(
        (targets.size(0), num_classes),
        off_val,
        dtype=torch.float32,
        device=targets.device
    )
    y_onehot.scatter_(1, targets.unsqueeze(1), on_val)
    return y_onehot


def pairwise_mixup_target(
        targets: torch.Tensor,
        pair_to: Dict[int, int],
        lam_list: List[float],
        *,
        num_classes: int,
        smoothing: float = 0.0,
) -> torch.Tensor:
    """Create soft targets that match the pixel‑level mixing performed.

    Args:
        targets: (B,) tensor of integer class indices.
        pair_to: Mapping of sample index to its mixed partner as returned by mix_batch_variable_size().
        lam_list: Per‑sample fractions of own pixels, also from the mixer.
        num_classes: Total number of classes in the dataset.
        smoothing: Label‑smoothing value in the range [0, 1).

    Returns:
        Tensor of shape (B, num_classes) whose rows sum to 1.
    """
    y_onehot = smoothed_sparse_target(targets, num_classes=num_classes, smoothing=smoothing)
    targets = y_onehot.clone()
    for i, j in pair_to.items():
        lam = lam_list[i]
        targets[i].mul_(lam).add_(y_onehot[j], alpha=1.0 - lam)

    return targets


class NaFlexMixup:
    """Callable wrapper that combines mixing and target generation."""

    def __init__(
            self,
            *,
            num_classes: int,
            mixup_alpha: float = 0.8,
            cutmix_alpha: float = 1.0,
            switch_prob: float = 0.5,
            prob: float = 1.0,
            local_shuffle: int = 4,
            label_smoothing: float = 0.0,
    ) -> None:
        """Configure the augmentation.

        Args:
            num_classes: Total number of classes.
            mixup_alpha: Beta α for Mixup. 0 disables Mixup.
            cutmix_alpha: Beta α for CutMix. 0 disables CutMix.
            switch_prob: Probability of selecting CutMix when both modes are enabled.
            prob: Probability of applying any mixing per batch.
            local_shuffle: Window size used to shuffle images after aspect sorting so pairings vary between epochs.
            smoothing: Label‑smoothing value. 0 disables smoothing.
        """
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob
        self.prob = prob
        self.local_shuffle = local_shuffle
        self.smoothing = label_smoothing

    def __call__(
            self,
            imgs: List[torch.Tensor],
            targets: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Apply the augmentation and generate matching targets.

        Args:
            imgs: List of already transformed images shaped (C, H, W).
            targets: Hard labels with shape (B,).

        Returns:
            mixed_imgs: List of mixed images in the same order and shapes as the input.
            targets: Soft‑label tensor shaped (B, num_classes) suitable for cross‑entropy with soft targets.
        """
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        if random.random() > self.prob:
            targets = smoothed_sparse_target(targets, num_classes=self.num_classes, smoothing=self.smoothing)
            return imgs, targets.unbind(0)

        mixed_imgs, lam_list, pair_to = mix_batch_variable_size(
            imgs,
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            switch_prob=self.switch_prob,
            local_shuffle=self.local_shuffle,
        )

        targets = pairwise_mixup_target(
            targets,
            pair_to,
            lam_list,
            num_classes=self.num_classes,
            smoothing=self.smoothing,
        )
        return mixed_imgs, targets.unbind(0)
