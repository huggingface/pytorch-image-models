import torch
from data.random_erasing import RandomErasingTorch


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
            fp16=False,
            random_erasing=True,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]):
        self.loader = loader
        self.fp16 = fp16
        self.random_erasing = random_erasing
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        if random_erasing:
            self.random_erasing = RandomErasingTorch(per_pixel=True)
        else:
            self.random_erasing = None

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half()
                else:
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
