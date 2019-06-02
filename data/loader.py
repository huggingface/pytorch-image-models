import torch.utils.data
from data.transforms import *
from data.distributed_sampler import OrderedDistributedSampler
from data.mixup import FastCollateMixup


def fast_collate(batch):
    targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    batch_size = len(targets)
    tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
    for i in range(batch_size):
        tensor[i] += torch.from_numpy(batch[i][0])

    return tensor, targets


class PrefetchLoader:

    def __init__(self,
            loader,
            rand_erase_prob=0.,
            rand_erase_mode='const',
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        if rand_erase_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=rand_erase_prob, mode=rand_erase_mode)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        curr_input = None

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float().sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield curr_input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            curr_input = next_input
            target = next_target

        yield curr_input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

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


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        rand_erase_prob=0.,
        rand_erase_mode='const',
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        tf_preprocessing=False,
):
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        from data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(is_training=is_training, size=img_size)
    else:
        if is_training:
            transform = transforms_imagenet_train(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std)
        else:
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                crop_pct=crop_pct)

    dataset.transform = transform

    sampler = None
    if distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=is_training,
    )
    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            rand_erase_prob=rand_erase_prob if is_training else 0.,
            rand_erase_mode=rand_erase_mode,
            mean=mean,
            std=std)

    return loader
