import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBinaryCrossEntropy(nn.Module):
    """ BCE using one-hot from dense targets w/ label smoothing
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(self, smoothing=0.1):
        super(DenseBinaryCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full(
            (target.size()[0], num_classes), off_value, device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        return self.bce(x, target)
