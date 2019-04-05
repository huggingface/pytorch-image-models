import torch
import torch.utils.data as tdata
from data.random_erasing import RandomErasingTorch
from data.transforms import *


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
            random_erasing=0.,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD):
        self.loader = loader
        self.random_erasing = random_erasing
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        if random_erasing:
            self.random_erasing = RandomErasingTorch(
                probability=random_erasing, per_pixel=False)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                next_input = next_input.sub_(self.mean).div_(self.std)
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


def create_loader(
        dataset,
        img_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        random_erasing=0.,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
):

    if is_training:
        transform = transforms_imagenet_train(
            img_size,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std)
    else:
        transform = transforms_imagenet_eval(
            img_size,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std)

    dataset.transform = transform

    sampler = None
    if distributed:
        sampler = tdata.distributed.DistributedSampler(dataset)

    loader = tdata.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=fast_collate if use_prefetcher else tdata.dataloader.default_collate,
    )
    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            random_erasing=random_erasing if is_training else 0.,
            mean=mean,
            std=std)

    return loader
