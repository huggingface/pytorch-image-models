"""Data loading utilities for timm engine.

Provides functions for creating datasets and data loaders with
augmentation pipelines for training and evaluation.
"""
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from timm.data import (
    AugMixDataset,
    FastCollateMixup,
    Mixup,
    MultiViewCollator,
    MultiViewTransform,
    create_dataset,
    create_loader,
    create_transform,
)

from .config import (
    AugmentConfig,
    DataConfig,
    LoaderConfig,
    MixupConfig,
    NaFlexConfig,
    TrainConfig,
)
from .device import DeviceEnv, is_primary

_logger = logging.getLogger(__name__)

# Try to import NaFlex components
try:
    from timm.data import NaFlexMixup, create_naflex_loader
    HAS_NAFLEX = True
except ImportError:
    HAS_NAFLEX = False


def create_train_dataset(
    cfg: TrainConfig,
    data_config: Dict[str, Any],
) -> Any:
    """Create training dataset.

    Args:
        cfg: Training configuration.
        data_config: Data configuration from model (input_size, etc.).

    Returns:
        Training dataset instance.
    """
    # Resolve input image mode
    if cfg.data.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = cfg.data.input_img_mode

    dataset = create_dataset(
        cfg.data.dataset,
        root=cfg.data.data_dir,
        split=cfg.data.train_split,
        is_training=True,
        class_map=cfg.data.class_map,
        download=cfg.data.dataset_download,
        batch_size=cfg.loader.batch_size,
        seed=cfg.misc.seed,
        repeats=cfg.scheduler.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=cfg.data.input_key,
        target_key=cfg.data.target_key,
        num_samples=cfg.data.train_num_samples,
        trust_remote_code=cfg.data.dataset_trust_remote_code,
    )

    return dataset


def create_eval_dataset(
    cfg: TrainConfig,
    data_config: Dict[str, Any],
) -> Optional[Any]:
    """Create evaluation dataset.

    Args:
        cfg: Training configuration.
        data_config: Data configuration from model (input_size, etc.).

    Returns:
        Evaluation dataset instance, or None if no val_split specified.
    """
    if not cfg.data.val_split:
        return None

    # Resolve input image mode
    if cfg.data.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = cfg.data.input_img_mode

    dataset = create_dataset(
        cfg.data.dataset,
        root=cfg.data.data_dir,
        split=cfg.data.val_split,
        is_training=False,
        class_map=cfg.data.class_map,
        download=cfg.data.dataset_download,
        batch_size=cfg.loader.batch_size,
        input_img_mode=input_img_mode,
        input_key=cfg.data.input_key,
        target_key=cfg.data.target_key,
        num_samples=cfg.data.val_num_samples,
        trust_remote_code=cfg.data.dataset_trust_remote_code,
    )

    return dataset


def create_train_loader(
    cfg: TrainConfig,
    data_config: Dict[str, Any],
    device_env: DeviceEnv,
    num_classes: int,
    model_patch_size: Optional[Tuple[int, int]] = None,
) -> Tuple[DataLoader, Optional[Callable], bool]:
    """Create training data loader with augmentation pipeline.

    This function handles:
    - Standard or NaFlex loader creation
    - Augmentation configuration
    - Mixup/CutMix setup
    - Distributed data loading

    Args:
        cfg: Training configuration.
        data_config: Data configuration from model (input_size, mean, std, etc.).
        device_env: Device environment.
        num_classes: Number of output classes (for mixup).
        model_patch_size: Model's patch size for NaFlex loader.

    Returns:
        Tuple of (loader, mixup_fn, naflex_mode) where:
        - loader: DataLoader instance
        - mixup_fn: Mixup/CutMix function (only for non-prefetcher mode)
        - naflex_mode: Whether NaFlex loader is being used

    Example::

        loader_train, mixup_fn, naflex_mode = create_train_loader(
            cfg, data_config, device_env, num_classes=1000
        )
    """
    # Create dataset
    dataset_train = create_train_dataset(cfg, data_config)

    # Resolve train interpolation
    train_interpolation = cfg.augment.train_interpolation
    if cfg.augment.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    # Common loader kwargs
    common_loader_kwargs = dict(
        mean=data_config['mean'],
        std=data_config['std'],
        pin_memory=cfg.loader.pin_mem,
        img_dtype=device_env.model_dtype or torch.float32,
        device=device_env.device,
        distributed=device_env.distributed,
        use_prefetcher=cfg.loader.prefetcher,
    )

    # Augmentation splits
    num_aug_splits = cfg.augment.aug_splits if cfg.augment.aug_splits > 1 else 0

    # Training loader kwargs
    train_loader_kwargs = dict(
        batch_size=cfg.loader.batch_size,
        is_training=True,
        no_aug=cfg.augment.no_aug,
        re_prob=cfg.augment.reprob,
        re_mode=cfg.augment.remode,
        re_count=cfg.augment.recount,
        re_split=cfg.augment.resplit,
        train_crop_mode=cfg.augment.train_crop_mode,
        scale=cfg.augment.scale,
        ratio=cfg.augment.ratio,
        hflip=cfg.augment.hflip,
        vflip=cfg.augment.vflip,
        color_jitter=cfg.augment.color_jitter,
        color_jitter_prob=cfg.augment.color_jitter_prob,
        grayscale_prob=cfg.augment.grayscale_prob,
        gaussian_blur_prob=cfg.augment.gaussian_blur_prob,
        auto_augment=cfg.augment.auto_augment,
        num_aug_repeats=cfg.augment.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        num_workers=cfg.loader.workers,
        worker_seeding=cfg.loader.worker_seeding,
    )

    # Mixup/CutMix setup
    mixup_fn = None
    mixup_active = (
        cfg.mixup.mixup > 0 or
        cfg.mixup.cutmix > 0 or
        cfg.mixup.cutmix_minmax is not None
    )

    mixup_args = {}
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=cfg.mixup.mixup,
            cutmix_alpha=cfg.mixup.cutmix,
            cutmix_minmax=cfg.mixup.cutmix_minmax,
            prob=cfg.mixup.mixup_prob,
            switch_prob=cfg.mixup.mixup_switch_prob,
            mode=cfg.mixup.mixup_mode,
            label_smoothing=cfg.mixup.smoothing,
            num_classes=num_classes,
        )

    naflex_mode = False

    if cfg.naflex.naflex_loader:
        assert HAS_NAFLEX, 'NaFlex loader not available'
        if is_primary(device_env):
            _logger.info('Using NaFlex loader')

        assert num_aug_splits <= 1, 'Augmentation splits not supported in NaFlex mode'

        naflex_mixup_fn = None
        if mixup_active:
            naflex_mixup_args = mixup_args.copy()
            naflex_mixup_args.pop('mode')  # not supported
            naflex_mixup_args.pop('cutmix_minmax')  # not supported
            naflex_mixup_fn = NaFlexMixup(**naflex_mixup_args)

        # Resolve patch size
        if model_patch_size is None:
            model_patch_size = (16, 16)
            if is_primary(device_env):
                _logger.warning(
                    f'Could not determine model patch size, using default: {model_patch_size}'
                )

        # Configure patch sizes
        patch_loader_kwargs = {}
        if cfg.naflex.naflex_patch_sizes:
            patch_loader_kwargs['patch_size_choices'] = cfg.naflex.naflex_patch_sizes
            if cfg.naflex.naflex_patch_size_probs:
                patch_loader_kwargs['patch_size_choice_probs'] = cfg.naflex.naflex_patch_size_probs
            if is_primary(device_env):
                _logger.info(f'Using variable patch sizes: {cfg.naflex.naflex_patch_sizes}')
        else:
            patch_loader_kwargs['patch_size'] = model_patch_size
            if is_primary(device_env):
                _logger.info(f'Using model patch size: {model_patch_size}')

        naflex_mode = True
        loader = create_naflex_loader(
            dataset=dataset_train,
            train_seq_lens=cfg.naflex.naflex_train_seq_lens,
            mixup_fn=naflex_mixup_fn,
            rank=device_env.rank,
            world_size=device_env.world_size,
            **patch_loader_kwargs,
            **common_loader_kwargs,
            **train_loader_kwargs,
        )
    else:
        # Standard loader with mixup
        collate_fn = None
        if mixup_active:
            if cfg.loader.prefetcher:
                assert not num_aug_splits, \
                    'Collate conflict: cannot use prefetcher mixup with aug splits'
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

        # Wrap dataset in AugMix helper if using splits
        if num_aug_splits > 1:
            dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

        loader = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            collate_fn=collate_fn,
            use_multi_epochs_loader=cfg.loader.use_multi_epochs_loader,
            **common_loader_kwargs,
            **train_loader_kwargs,
        )

    return loader, mixup_fn, naflex_mode


def create_eval_loader(
    cfg: TrainConfig,
    data_config: Dict[str, Any],
    device_env: DeviceEnv,
    model_patch_size: Optional[Tuple[int, int]] = None,
) -> Optional[DataLoader]:
    """Create evaluation data loader.

    Args:
        cfg: Training configuration.
        data_config: Data configuration from model.
        device_env: Device environment.
        model_patch_size: Model's patch size for NaFlex loader.

    Returns:
        DataLoader instance, or None if no val_split specified.
    """
    # Create dataset
    dataset_eval = create_eval_dataset(cfg, data_config)
    if dataset_eval is None:
        return None

    # Common loader kwargs
    common_loader_kwargs = dict(
        mean=data_config['mean'],
        std=data_config['std'],
        pin_memory=cfg.loader.pin_mem,
        img_dtype=device_env.model_dtype or torch.float32,
        device=device_env.device,
        distributed=device_env.distributed,
        use_prefetcher=cfg.loader.prefetcher,
    )

    # Adjust workers for distributed TFDS/WDS
    eval_workers = cfg.loader.workers
    if device_env.distributed and ('tfds' in cfg.data.dataset or 'wds' in cfg.data.dataset):
        eval_workers = min(2, cfg.loader.workers)

    eval_loader_kwargs = dict(
        batch_size=cfg.loader.validation_batch_size or cfg.loader.batch_size,
        is_training=False,
        interpolation=data_config['interpolation'],
        num_workers=eval_workers,
        crop_pct=data_config['crop_pct'],
    )

    if cfg.naflex.naflex_loader:
        assert HAS_NAFLEX, 'NaFlex loader not available'

        if model_patch_size is None:
            model_patch_size = (16, 16)

        loader = create_naflex_loader(
            dataset=dataset_eval,
            patch_size=model_patch_size,
            max_seq_len=cfg.naflex.naflex_max_seq_len,
            **common_loader_kwargs,
            **eval_loader_kwargs,
        )
    else:
        loader = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            **common_loader_kwargs,
            **eval_loader_kwargs,
        )

    return loader


def create_multiview_train_loader(
    cfg: TrainConfig,
    data_config: Dict[str, Any],
    device_env: DeviceEnv,
    num_views: int = 2,
) -> DataLoader:
    """Create training data loader for multi-view SSL methods.

    Creates a loader that returns batches of shape [B, V, C, H, W] where
    V is the number of augmented views per image. Each view is created
    by applying the same augmentation pipeline with different random seeds.

    Used for methods like LeJEPA, DINO, SimCLR that require multiple views.

    Args:
        cfg: Training configuration.
        data_config: Data configuration from model (input_size, mean, std, etc.).
        device_env: Device environment.
        num_views: Number of augmented views per image.

    Returns:
        DataLoader that yields (images, targets) where images is [B, V, C, H, W].

    Example::

        loader = create_multiview_train_loader(cfg, data_config, device_env, num_views=2)
        for images, targets in loader:
            # images: [B, V, C, H, W]
            # targets: [B]
            pass
    """
    # Create dataset (without transform - we'll set it below)
    dataset_train = create_train_dataset(cfg, data_config)

    # Resolve train interpolation
    train_interpolation = cfg.augment.train_interpolation
    if cfg.augment.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    # Create transform kwargs (same for all views)
    input_size = data_config['input_size']
    transform_kwargs = dict(
        input_size=input_size,
        is_training=True,
        no_aug=cfg.augment.no_aug,
        train_crop_mode=cfg.augment.train_crop_mode,
        scale=cfg.augment.scale,
        ratio=cfg.augment.ratio,
        hflip=cfg.augment.hflip,
        vflip=cfg.augment.vflip,
        color_jitter=cfg.augment.color_jitter,
        color_jitter_prob=cfg.augment.color_jitter_prob,
        grayscale_prob=cfg.augment.grayscale_prob,
        gaussian_blur_prob=cfg.augment.gaussian_blur_prob,
        auto_augment=cfg.augment.auto_augment,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        re_prob=cfg.augment.reprob,
        re_mode=cfg.augment.remode,
        re_count=cfg.augment.recount,
        use_prefetcher=False,  # Multi-view doesn't use prefetcher
    )

    # Create multiple transforms (same config, different random seeds)
    transforms = [create_transform(**transform_kwargs) for _ in range(num_views)]
    dataset_train.transform = MultiViewTransform(transforms)

    # Sampler setup
    sampler = None
    if device_env.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

    # Create loader with multi-view collator
    loader = DataLoader(
        dataset_train,
        batch_size=cfg.loader.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.loader.workers,
        collate_fn=MultiViewCollator(),
        pin_memory=cfg.loader.pin_mem,
        drop_last=True,
        persistent_workers=cfg.loader.workers > 0,
    )

    if is_primary(device_env):
        _logger.info(
            f'Created multi-view train loader: {num_views} views, '
            f'batch_size={cfg.loader.batch_size}, input_size={input_size}'
        )

    return loader


def create_multiview_eval_loader(
    cfg: TrainConfig,
    data_config: Dict[str, Any],
    device_env: DeviceEnv,
    num_views: int = 2,
) -> Optional[DataLoader]:
    """Create eval data loader for multi-view SSL methods.

    Creates a loader that returns batches of shape [B, V, C, H, W] for
    validation/eval. Uses deterministic transforms (center crop, no augmentation)
    applied V times to create identical views for loss computation.

    Args:
        cfg: Training configuration.
        data_config: Data configuration from model (input_size, mean, std, etc.).
        device_env: Device environment.
        num_views: Number of views per image (should match training).

    Returns:
        DataLoader that yields (images, targets) where images is [B, V, C, H, W],
        or None if no val_split specified.
    """
    dataset_eval = create_eval_dataset(cfg, data_config)
    if dataset_eval is None:
        return None

    input_size = data_config['input_size']

    # Create eval transform (no augmentation, center crop)
    transform_kwargs = dict(
        input_size=input_size,
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config['crop_pct'],
        use_prefetcher=False,
    )

    # Create same transform V times (deterministic, so all views are identical)
    # This is intentional for eval - we want consistent views
    transforms = [create_transform(**transform_kwargs) for _ in range(num_views)]
    dataset_eval.transform = MultiViewTransform(transforms)

    # Sampler setup
    sampler = None
    if device_env.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval, shuffle=False)

    batch_size = cfg.loader.validation_batch_size or cfg.loader.batch_size

    loader = DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.loader.workers,
        collate_fn=MultiViewCollator(),
        pin_memory=cfg.loader.pin_mem,
        drop_last=False,
        persistent_workers=cfg.loader.workers > 0,
    )

    if is_primary(device_env):
        _logger.info(
            f'Created multi-view eval loader: {num_views} views, '
            f'batch_size={batch_size}, input_size={input_size}'
        )

    return loader


def get_num_classes_from_dataset(dataset: Any) -> int:
    """Get number of classes from dataset.

    Args:
        dataset: Dataset instance.

    Returns:
        Number of classes.
    """
    if hasattr(dataset, 'classes'):
        return len(dataset.classes)
    elif hasattr(dataset, 'num_classes'):
        return dataset.num_classes
    else:
        raise ValueError('Cannot determine number of classes from dataset')
