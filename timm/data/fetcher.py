import torch

from .constants import *
from .random_erasing import RandomErasing
from. mixup import FastCollateMixup


class FetcherXla:
    def __init__(self):
        pass


class Fetcher:

    def __init__(self,
                 loader,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD,
                 device=None,
                 dtype=None,
                 re_prob=0.,
                 re_mode='const',
                 re_count=1,
                 re_num_splits=0):
        self.loader = loader
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        self.mean = torch.tensor([x * 255 for x in mean], dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std], dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        if re_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device=device)
        else:
            self.random_erasing = None

    def __iter__(self):
        for sample, target in self.loader:
            sample = sample.to(device=self.device, dtype=self.dtype).sub_(self.mean).div_(self.std)
            target = target.to(device=self.device)
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