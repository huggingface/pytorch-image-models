""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2019 Ross Wightman
"""

from typing import Optional, Callable

import numpy as np
import torch.utils.data

from timm.bits import DeviceEnv
from .collate import fast_collate
from .config import PreprocessCfg, MixupCfg
from .distributed_sampler import OrderedDistributedSampler
from .fetcher import Fetcher
from .mixup import FastCollateMixup
from .prefetcher_cuda import PrefetcherCuda
from .transforms_factory import create_transform_v2


def _worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    np.random.seed(worker_info.seed % (2**32-1))


def create_loader_v2(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        is_training: bool = False,
        dev_env: Optional[DeviceEnv] = None,
        pp_cfg: PreprocessCfg = PreprocessCfg(),
        mix_cfg: MixupCfg = None,
        create_transform: bool = True,
        normalize_in_transform: bool = True,
        separate_transform: bool = False,
        num_workers: int = 1,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        use_multi_epochs_loader: bool = False,
        persistent_workers: bool = True,
):
    """
    
    Args:
        dataset: 
        batch_size: 
        is_training: 
        dev_env:
        pp_cfg: 
        mix_cfg:
        create_transform:
        normalize_in_transform:
        separate_transform:
        num_workers: 
        collate_fn: 
        pin_memory: 
        use_multi_epochs_loader: 
        persistent_workers: 

    Returns:

    """
    if dev_env is None:
        dev_env = DeviceEnv.instance()

    if create_transform:
        dataset.transform = create_transform_v2(
            cfg=pp_cfg,
            is_training=is_training,
            normalize=normalize_in_transform,
            separate=separate_transform,
        )

    sampler = None
    if dev_env.distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=dev_env.world_size, rank=dev_env.global_rank)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(
                dataset, num_replicas=dev_env.world_size, rank=dev_env.global_rank)

    if collate_fn is None:
        if mix_cfg is not None and mix_cfg.prob > 0:
            collate_fn = FastCollateMixup(
                mixup_alpha=mix_cfg.mixup_alpha,
                cutmix_alpha=mix_cfg.cutmix_alpha,
                cutmix_minmax=mix_cfg.cutmix_minmax,
                prob=mix_cfg.prob,
                switch_prob=mix_cfg.switch_prob,
                mode=mix_cfg.mode,
                correct_lam=mix_cfg.correct_lam,
                label_smoothing=mix_cfg.label_smoothing,
                num_classes=mix_cfg.num_classes,
            )
        else:
            collate_fn = fast_collate

    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=_worker_init,
        persistent_workers=persistent_workers)
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    fetcher_kwargs = dict(
        normalize=not normalize_in_transform,
        mean=pp_cfg.mean,
        std=pp_cfg.std,
    )
    if not normalize_in_transform and is_training and pp_cfg.aug is not None:
        # If normalization can be done in the prefetcher, random erasing is done there too
        # NOTE RandomErasing does not work well in XLA so normalize_in_transform will be True
        fetcher_kwargs.update(dict(
            re_prob=pp_cfg.aug.re_prob,
            re_mode=pp_cfg.aug.re_mode,
            re_count=pp_cfg.aug.re_count,
            num_aug_splits=pp_cfg.aug.num_aug_splits,
        ))
    if dev_env.type_cuda:
        loader = PrefetcherCuda(loader, **fetcher_kwargs)
    else:
        loader = Fetcher(loader, device=dev_env.device, **fetcher_kwargs)

    return loader


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
