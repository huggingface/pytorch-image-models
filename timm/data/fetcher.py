import torch

from .constants import *
from .random_erasing import RandomErasing
from .mixup import FastCollateMixup


class FetcherXla:
    def __init__(self):
        pass


class Fetcher:

    def __init__(
            self,
            loader,
            device: torch.device,
            dtype=torch.float32,
            normalize=True,
            normalize_shape=(1, 3, 1, 1),
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            re_prob=0.,
            re_mode='const',
            re_count=1,
            num_aug_splits=0,
            use_mp_loader=False,
    ):
        self.loader = loader
        self.device = torch.device(device)
        self.dtype = dtype
        if normalize:
            self.mean = torch.tensor(
                [x * 255 for x in mean], dtype=self.dtype, device=self.device).view(normalize_shape)
            self.std = torch.tensor(
                [x * 255 for x in std], dtype=self.dtype, device=self.device).view(normalize_shape)
        else:
            self.mean = None
            self.std = None
        if re_prob > 0.:
            # NOTE RandomErasing shouldn't be used here w/ XLA devices
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, count=re_count, num_splits=num_aug_splits)
        else:
            self.random_erasing = None
        self.use_mp_loader = use_mp_loader
        if use_mp_loader:
            # FIXME testing for TPU use
            import torch_xla.distributed.parallel_loader as pl
            self._loader = pl.MpDeviceLoader(loader, device)
        else:
            self._loader = loader

    def __iter__(self):
        for sample, target in self._loader:
            if not self.use_mp_loader:
                sample = sample.to(device=self.device)
                target = target.to(device=self.device)
            sample = sample.to(dtype=self.dtype)
            if self.mean is not None:
                sample.sub_(self.mean).div_(self.std)
            if self.random_erasing is not None:
                sample = self.random_erasing(sample)
            yield sample, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x