import torch.cuda

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .mixup import FastCollateMixup
from .random_erasing import RandomErasing


class PrefetcherCuda:

    def __init__(
            self,
            loader,
            device: torch.device = torch.device('cuda'),
            dtype=torch.float32,
            normalize=True,
            normalize_shape=(1, 3, 1, 1),
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            re_prob=0.,
            re_mode='const',
            re_count=1,
            num_aug_splits=0,
    ):
        self.loader = loader
        self.device = device
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
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, count=re_count, num_splits=num_aug_splits)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_input = next_input.to(dtype=self.dtype)
                if self.mean is not None:
                    next_input.sub_(self.mean).div_(self.std)
                next_target = next_target.to(device=self.device, non_blocking=True)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

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
