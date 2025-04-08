import math
from contextlib import suppress
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .loader import _worker_init
from .naflex_dataset import VariableSeqMapWrapper
from .transforms_factory import create_transform


class NaFlexCollator:
    """Custom collator for batching NaFlex-style variable-resolution images."""

    def __init__(
            self,
            patch_size=16,
            max_seq_len=None,
    ):
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len or 576  # Default ViT-B/16 sequence length (577 = 24*24)

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (patch_dict, target)

        Returns:
            A tuple of (input_dict, targets) where input_dict contains:
                - patches: Padded tensor of patches
                - patch_coord: Coordinates for each patch (y, x)
                - patch_valid: Valid indicators
        """
        assert isinstance(batch[0], tuple)
        batch_size = len(batch)

        # Resize to final size based on seq_len and patchify

        # Extract targets
        targets = torch.tensor([item[1] for item in batch], dtype=torch.int64)

        # Get patch dictionaries
        patch_dicts = [item[0] for item in batch]

        # If we have a maximum sequence length constraint, ensure we don't exceed it
        if self.max_seq_len is not None:
            max_patches = self.max_seq_len
        else:
            # Find the maximum number of patches in this batch
            max_patches = max(item['patches'].shape[0] for item in patch_dicts)

        # Get patch dimensionality
        patch_dim = patch_dicts[0]['patches'].shape[1]

        # Prepare tensors for the batch
        patches = torch.zeros((batch_size, max_patches, patch_dim), dtype=torch.float32)
        patch_coord = torch.zeros((batch_size, max_patches, 2), dtype=torch.int64)  # [B, N, 2] for (y, x)
        patch_valid = torch.zeros((batch_size, max_patches), dtype=torch.bool)

        # Fill in the tensors
        for i, patch_dict in enumerate(patch_dicts):
            num_patches = min(patch_dict['patches'].shape[0], max_patches)

            patches[i, :num_patches] = patch_dict['patches'][:num_patches]
            patch_coord[i, :num_patches] = patch_dict['patch_coord'][:num_patches]
            patch_valid[i, :num_patches] = patch_dict['patch_valid'][:num_patches]

        return {
            'patches': patches,
            'patch_coord': patch_coord,
            'patch_valid': patch_valid,
            'seq_len': max_patches,
        }, targets


class NaFlexPrefetchLoader:
    """Data prefetcher for NaFlex format which normalizes patches."""

    def __init__(
            self,
            loader,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            img_dtype=torch.float32,
            device=torch.device('cuda')
        ):
        self.loader = loader
        self.device = device
        self.img_dtype = img_dtype or torch.float32

        # Create mean/std tensors for normalization (will be applied to patches)
        self.mean = torch.tensor([x * 255 for x in mean], device=device, dtype=self.img_dtype).view(1, 1, 3)
        self.std = torch.tensor([x * 255 for x in std], device=device, dtype=self.img_dtype).view(1, 1, 3)

        # Check for CUDA/NPU availability
        self.is_cuda = device.type == 'cuda' and torch.cuda.is_available()
        self.is_npu = device.type == 'npu' and torch.npu.is_available()

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        elif self.is_npu:
            stream = torch.npu.Stream()
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

                # Normalize patch values (assuming patches are in format [B, N, P*P*C])
                batch_size, num_patches, patch_pixels = next_input_dict['patches'].shape
                patches = next_input_dict['patches'].view(batch_size, -1, 3) # to [B*N, P*P, C] for normalization
                patches = patches.sub(self.mean).div(self.std)

                # Reshape back
                next_input_dict['patches'] = patches.reshape(batch_size, num_patches, patch_pixels)

            if not first:
                yield input_dict, target
            else:
                first = False

            if stream is not None:
                if self.is_cuda:
                    torch.cuda.current_stream().wait_stream(stream)
                elif self.is_npu:
                    torch.npu.current_stream().wait_stream(stream)

            input_dict = next_input_dict
            target = next_target

        yield input_dict, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_naflex_loader(
        dataset,
        patch_size: Union[Tuple[int, int], int] = 16,
        train_seq_lens: List[int] = (128, 256, 576, 784, 1024),  # Training sequence lengths
        max_seq_len: int = 576,  # Fixed sequence length for validation
        batch_size: int = 32,  # Used for max_seq_len and max(train_seq_lens)
        is_training: bool = False,

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
    ):
    """Create a data loader with dynamic sequence length sampling for training."""

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

        naflex_dataset = VariableSeqMapWrapper(
            dataset,
            transform_factory=transform_factory,
            patch_size=patch_size,
            seq_lens=train_seq_lens,
            max_tokens_per_batch=max_tokens_per_batch,
            seed=seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            epoch=epoch,
        )

        # NOTE: Collation is handled by the dataset wrapper for training
        # Create the collator (handles fixed-size collation)
        # collate_fn = NaFlexCollator(
        #     patch_size=patch_size,
        #     max_seq_len=max(seq_lens) + 1,  # +1 for class token
        #     use_prefetcher=use_prefetcher
        # )

        loader = torch.utils.data.DataLoader(
            naflex_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            sampler=None,
            #collate_fn=collate_fn,
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
        collate_fn = NaFlexCollator(
            patch_size=patch_size,
            max_seq_len=max_seq_len,
        )

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
