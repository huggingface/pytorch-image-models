"""NaFlex data loader for dynamic sequence length training.

This module provides a specialized data loader for Vision Transformer models that supports:
- Dynamic sequence length sampling during training for improved efficiency
- Variable patch size training with probabilistic selection
- Patch-level random erasing augmentation
- Efficient GPU prefetching with normalization

Hacked together by / Copyright 2025, Ross Wightman, Hugging Face
"""

import math
from contextlib import suppress
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union


import torch

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .loader import _worker_init, adapt_to_chs
from .naflex_dataset import NaFlexMapDatasetWrapper, NaFlexCollator
from .naflex_random_erasing import PatchRandomErasing
from .transforms_factory import create_transform


class NaFlexPrefetchLoader:
    """Data prefetcher for NaFlex format which normalizes patches."""

    def __init__(
            self,
            loader: torch.utils.data.DataLoader,
            mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
            std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
            channels: int = 3,
            device: torch.device = torch.device('cuda'),
            img_dtype: Optional[torch.dtype] = None,
            re_prob: float = 0.,
            re_mode: str = 'const',
            re_count: int = 1,
            re_num_splits: int = 0,
        ) -> None:
        """Initialize NaFlexPrefetchLoader.

        Args:
            loader: DataLoader to prefetch from.
            mean: Mean values for normalization.
            std: Standard deviation values for normalization.
            channels: Number of image channels.
            device: Device to move tensors to.
            img_dtype: Data type for image tensors.
            re_prob: Random erasing probability.
            re_mode: Random erasing mode.
            re_count: Maximum number of erasing rectangles.
            re_num_splits: Number of augmentation splits.
        """
        self.loader = loader
        self.device = device
        self.img_dtype = img_dtype or torch.float32

        # Create mean/std tensors for normalization (will be applied to patches)
        mean = adapt_to_chs(mean, channels)
        std = adapt_to_chs(std, channels)
        normalization_shape = (1, 1, channels)
        self.channels = channels
        self.mean = torch.tensor(
            [x * 255 for x in mean], device=device, dtype=self.img_dtype).view(normalization_shape)
        self.std = torch.tensor(
            [x * 255 for x in std], device=device, dtype=self.img_dtype).view(normalization_shape)

        if re_prob > 0.:
            self.random_erasing = PatchRandomErasing(
                erase_prob=re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device=device,
            )
        else:
            self.random_erasing = None

        # Check for CUDA/NPU availability
        self.is_cuda = device.type == 'cuda' and torch.cuda.is_available()
        self.is_npu = device.type == 'npu' and torch.npu.is_available()

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """Iterate through the loader with prefetching and normalization.

        Yields:
            Tuple of (input_dict, targets) with normalized patches.
        """
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream(device=self.device)
            stream_context = partial(torch.cuda.stream, stream=stream)
        elif self.is_npu:
            stream = torch.npu.Stream(device=self.device)
            stream_context = partial(torch.npu.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_input_dict, next_target in self.loader:
            with stream_context():
                # Move all tensors in input_dict to device
                for k, v in next_input_dict.items():
                    if isinstance(v, torch.Tensor):
                        dtype = self.img_dtype if k == 'patches' else None
                        next_input_dict[k] = next_input_dict[k].to(
                            device=self.device,
                            non_blocking=True,
                            dtype=dtype,
                        )

                next_target = next_target.to(device=self.device, non_blocking=True)

                # Normalize patch values - handle both [B, N, P*P*C] and [B, N, Ph, Pw, C] formats
                patches_tensor = next_input_dict['patches']
                original_shape = patches_tensor.shape

                if patches_tensor.ndim == 3:
                    # Format: [B, N, P*P*C] - flattened patches
                    batch_size, num_patches, patch_pixels = original_shape
                    # To [B*N, P*P, C] for normalization and erasing
                    patches = patches_tensor.view(batch_size, num_patches, -1, self.channels)
                elif patches_tensor.ndim == 5:
                    # Format: [B, N, Ph, Pw, C] - unflattened patches (variable patch size mode)
                    batch_size, num_patches, patch_h, patch_w, channels = original_shape
                    assert channels == self.channels, f"Expected {self.channels} channels, got {channels}"
                    # To [B*N, Ph*Pw, C] for normalization and erasing
                    patches = patches_tensor.view(batch_size, num_patches, -1, self.channels)
                else:
                    raise ValueError(f"Unexpected patches tensor dimensions: {patches_tensor.ndim}. Expected 3 or 5.")

                # Apply normalization
                patches = patches.sub(self.mean).div(self.std)

                if self.random_erasing is not None:
                    patches = self.random_erasing(
                        patches,
                        patch_coord=next_input_dict['patch_coord'],
                        patch_valid=next_input_dict.get('patch_valid', None),
                    )

                # Reshape back to original format
                next_input_dict['patches'] = patches.view(original_shape)

            if not first:
                yield input_dict, target
            else:
                first = False

            if stream is not None:
                if self.is_cuda:
                    torch.cuda.current_stream(device=self.device).wait_stream(stream)
                elif self.is_npu:
                    torch.npu.current_stream(device=self.device).wait_stream(stream)

            input_dict = next_input_dict
            target = next_target

        yield input_dict, target

    def __len__(self) -> int:
        """Get length of underlying loader.

        Returns:
            Number of batches in the loader.
        """
        return len(self.loader)

    @property
    def sampler(self):
        """Get sampler from underlying loader.

        Returns:
            Sampler from the underlying DataLoader.
        """
        return self.loader.sampler

    @property
    def dataset(self):
        """Get dataset from underlying loader.

        Returns:
            Dataset from the underlying DataLoader.
        """
        return self.loader.dataset


def create_naflex_loader(
        dataset,
        patch_size: Optional[Union[Tuple[int, int], int]] = None,
        patch_size_choices: Optional[List[int]] = None,
        patch_size_choice_probs: Optional[List[float]] = None,
        train_seq_lens: Tuple[int, ...] = (128, 256, 576, 784, 1024),
        max_seq_len: int = 576,
        batch_size: int = 32,
        is_training: bool = False,
        mixup_fn: Optional[Callable] = None,

        no_aug: bool = False,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_split: bool = False,
        train_crop_mode: Optional[str] = None,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
        hflip: float = 0.5,
        vflip: float = 0.,
        color_jitter: float = 0.4,
        color_jitter_prob: Optional[float] = None,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        num_aug_repeats: int = 0,
        num_aug_splits: int = 0,
        interpolation: str = 'bilinear',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        crop_pct: Optional[float] = None,
        crop_mode: Optional[str] = None,
        crop_border_pixels: Optional[int] = None,

        num_workers: int = 4,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        epoch: int = 0,
        use_prefetcher: bool = True,
        pin_memory: bool = True,
        img_dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = torch.device('cuda'),
        persistent_workers: bool = True,
        worker_seeding: str = 'all',
    ) -> Union[torch.utils.data.DataLoader, NaFlexPrefetchLoader]:
    """Create a data loader with dynamic sequence length sampling for training.

    Args:
        dataset: Dataset to load from.
        patch_size: Single patch size to use.
        patch_size_choices: List of patch sizes for variable patch size training.
        patch_size_choice_probs: Probabilities for each patch size choice.
        train_seq_lens: Training sequence lengths for dynamic batching.
        max_seq_len: Fixed sequence length for validation.
        batch_size: Batch size for validation and max training sequence length.
        is_training: Whether this is for training (enables dynamic batching).
        mixup_fn: Optional mixup function.
        no_aug: Disable augmentation.
        re_prob: Random erasing probability.
        re_mode: Random erasing mode.
        re_count: Maximum number of erasing rectangles.
        re_split: Random erasing split flag.
        train_crop_mode: Training crop mode.
        scale: Scale range for random resize crop.
        ratio: Aspect ratio range for random resize crop.
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Color jitter factor.
        color_jitter_prob: Color jitter probability.
        grayscale_prob: Grayscale conversion probability.
        gaussian_blur_prob: Gaussian blur probability.
        auto_augment: AutoAugment policy.
        num_aug_repeats: Number of augmentation repeats.
        num_aug_splits: Number of augmentation splits.
        interpolation: Interpolation method.
        mean: Normalization mean values.
        std: Normalization standard deviation values.
        crop_pct: Crop percentage for validation.
        crop_mode: Crop mode.
        crop_border_pixels: Crop border pixels.
        num_workers: Number of data loading workers.
        distributed: Whether using distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        seed: Random seed.
        epoch: Starting epoch.
        use_prefetcher: Whether to use prefetching.
        pin_memory: Whether to pin memory.
        img_dtype: Image data type.
        device: Device to move tensors to.
        persistent_workers: Whether to use persistent workers.
        worker_seeding: Worker seeding mode.

    Returns:
        DataLoader or NaFlexPrefetchLoader instance.
    """

    if is_training:
        # For training, use the dynamic sequence length mechanism
        assert num_aug_repeats == 0, 'Augmentation repeats not currently supported in NaFlex loader'

        transform_factory = partial(
            create_transform,
            is_training=True,
            no_aug=no_aug,
            train_crop_mode=train_crop_mode,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            color_jitter_prob=color_jitter_prob,
            grayscale_prob=grayscale_prob,
            gaussian_blur_prob=gaussian_blur_prob,
            auto_augment=auto_augment,
            interpolation=interpolation,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
            crop_mode=crop_mode,
            crop_border_pixels=crop_border_pixels,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            use_prefetcher=use_prefetcher,
            naflex=True,
        )

        max_train_seq_len = max(train_seq_lens)
        max_tokens_per_batch = batch_size * max_train_seq_len

        if isinstance(dataset, torch.utils.data.IterableDataset):
            assert False, "IterableDataset Wrapper is a WIP"

        naflex_dataset = NaFlexMapDatasetWrapper(
            dataset,
            transform_factory=transform_factory,
            patch_size=patch_size,
            patch_size_choices=patch_size_choices,
            patch_size_choice_probs=patch_size_choice_probs,
            seq_lens=train_seq_lens,
            max_tokens_per_batch=max_tokens_per_batch,
            mixup_fn=mixup_fn,
            seed=seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            epoch=epoch,
        )

        # NOTE: Collation is handled by the dataset wrapper for training
        loader = torch.utils.data.DataLoader(
            naflex_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            sampler=None,
            pin_memory=pin_memory,
            worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
            persistent_workers=persistent_workers
        )

        if use_prefetcher:
            loader = NaFlexPrefetchLoader(
                loader,
                mean=mean,
                std=std,
                img_dtype=img_dtype,
                device=device,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
            )

    else:
        # For validation, use fixed sequence length (unchanged)
        dataset.transform = create_transform(
            is_training=False,
            interpolation=interpolation,
            mean=mean,
            std=std,
            # FIXME add crop args when sequence transforms support crop modes
            use_prefetcher=use_prefetcher,
            naflex=True,
            patch_size=patch_size,
            max_seq_len=max_seq_len,
            patchify=True,
        )

        # Create the collator
        collate_fn = NaFlexCollator(max_seq_len=max_seq_len)

        # Handle distributed training
        sampler = None
        if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
            # For validation, use OrderedDistributedSampler
            from timm.data.distributed_sampler import OrderedDistributedSampler
            sampler = OrderedDistributedSampler(dataset)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
        )

        if use_prefetcher:
            loader = NaFlexPrefetchLoader(
                loader,
                mean=mean,
                std=std,
                img_dtype=img_dtype,
                device=device,
            )

    return loader
