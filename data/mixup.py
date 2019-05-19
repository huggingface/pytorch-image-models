import numpy as np
import torch


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return lam*y1 + (1. - lam)*y2


class FastCollateMixup:

    def __init__(self, mixup_alpha=1., label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mixup_enabled = True

    def __call__(self, batch):
        batch_size = len(batch)
        lam = 1.
        if self.mixup_enabled:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        target = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device='cpu')

        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            mixed = batch[i][0].astype(np.float32) * lam + \
                    batch[batch_size - i - 1][0].astype(np.float32) * (1 - lam)
            np.round(mixed, out=mixed)
            tensor[i] += torch.from_numpy(mixed.astype(np.uint8))

        return tensor, target
