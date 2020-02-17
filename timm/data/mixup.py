""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by Ross Wightman
"""

import numpy as np
import torch
import math
from enum import IntEnum


class MixupMode(IntEnum):
    MIXUP = 0
    CUTMIX = 1
    RANDOM = 2

    @classmethod
    def from_str(cls, value):
        return cls[value.upper()]


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


def mixup_batch(input, target, alpha=0.2, num_classes=1000, smoothing=0.1, disable=False):
    lam = 1.
    if not disable:
        lam = np.random.beta(alpha, alpha)
    input = input.mul(lam).add_(1 - lam, input.flip(0))
    target = mixup_target(target, num_classes, lam, smoothing)
    return input, target


def rand_bbox(size, ratio):
    H, W = size[-2:]
    ratio = max(min(ratio, 0.8), 0.2)
    cut_h, cut_w = int(H * ratio), int(W * ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    yl, yh = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    xl, xh = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    return yl, yh, xl, xh


def cutmix_batch(input, target, alpha=0.2, num_classes=1000, smoothing=0.1, disable=False):
    lam = 1.
    if not disable:
        lam = np.random.beta(alpha, alpha)
    if lam != 1:
        ratio = math.sqrt(1. - lam)
        yl, yh, xl, xh = rand_bbox(input.size(), ratio)
        input[:, :, yl:yh, xl:xh] = input.flip(0)[:, :, yl:yh, xl:xh]
    target = mixup_target(target, num_classes, lam, smoothing)
    return input, target


def _resolve_mode(mode):
    mode = MixupMode.from_str(mode) if isinstance(mode, str) else mode
    if mode == MixupMode.RANDOM:
        mode = MixupMode(np.random.rand() > 0.5)
    return mode # will be one of cutmix or mixup


def mix_batch(
        input, target, alpha=0.2, num_classes=1000, smoothing=0.1, disable=False, mode=MixupMode.MIXUP):
    mode = _resolve_mode(mode)
    if mode == MixupMode.CUTMIX:
        return mixup_batch(input, target, alpha, num_classes, smoothing, disable)
    else:
        return cutmix_batch(input, target, alpha, num_classes, smoothing, disable)


class FastCollateMixup:
    """Fast Collate Mixup that applies different params to each element + flipped pair

    NOTE once experiments are done, one of the three variants will remain with this class name
    """
    def __init__(self, mixup_alpha=1., label_smoothing=0.1, num_classes=1000, mode=MixupMode.MIXUP):
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = MixupMode.from_str(mode) if isinstance(mode, str) else mode
        self.mixup_enabled = True
        self.correct_lam = False  # correct lambda based on clipped area for cutmix

    def _do_mix(self, tensor, batch):
        batch_size = len(batch)
        lam_out = torch.ones(batch_size)
        for i in range(batch_size//2):
            j = batch_size - i - 1
            lam = 1.
            if self.mixup_enabled:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            if _resolve_mode(self.mode) == MixupMode.CUTMIX:
                mixed_i, mixed_j = batch[i][0].astype(np.float32), batch[j][0].astype(np.float32)
                ratio = math.sqrt(1. - lam)
                if lam != 1:
                    yl, yh, xl, xh = rand_bbox(tensor.size(), ratio)
                    mixed_i[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh].astype(np.float32)
                    mixed_j[:, yl:yh, xl:xh] = batch[i][0][:, yl:yh, xl:xh].astype(np.float32)
                    if self.correct_lam:
                        lam_corrected = (yh - yl) * (xh - xl) / (tensor.shape[-2] * tensor.shape[-1])
                        lam_out[i] -= lam_corrected
                        lam_out[j] -= lam_corrected
                    else:
                        lam_out[i] = lam
                        lam_out[j] = lam
            else:
                mixed_i = batch[i][0].astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                mixed_j = batch[j][0].astype(np.float32) * lam + batch[i][0].astype(np.float32) * (1 - lam)
                lam_out[i] = lam
                lam_out[j] = lam
            np.round(mixed_i, out=mixed_i)
            np.round(mixed_j, out=mixed_j)
            tensor[i] += torch.from_numpy(mixed_i.astype(np.uint8))
            tensor[j] += torch.from_numpy(mixed_j.astype(np.uint8))
        return lam_out

    def __call__(self, batch):
        batch_size = len(batch)
        assert batch_size % 2 == 0, 'Batch size should be even when using this'
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        lam = self._do_mix(tensor, batch)
        target = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        target = mixup_target(target, self.num_classes, lam.unsqueeze(1), self.label_smoothing, device='cpu')

        return tensor, target


class FastCollateMixupElementwise(FastCollateMixup):
    """Fast Collate Mixup that applies different params to each batch element

    NOTE this is for experimentation, may remove at some point
    """
    def __init__(self, mixup_alpha=1., label_smoothing=0.1, num_classes=1000, mode=MixupMode.MIXUP):
        super(FastCollateMixupElementwise, self).__init__(mixup_alpha, label_smoothing, num_classes, mode)

    def _do_mix(self, tensor, batch):
        batch_size = len(batch)
        lam_out = torch.ones(batch_size)
        for i in range(batch_size):
            lam = 1.
            if self.mixup_enabled:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            if _resolve_mode(self.mode) == MixupMode.CUTMIX:
                mixed = batch[i][0].astype(np.float32)
                ratio = math.sqrt(1. - lam)
                if lam != 1:
                    yl, yh, xl, xh = rand_bbox(tensor.size(), ratio)
                    mixed[:, yl:yh, xl:xh] = batch[batch_size - i - 1][0][:, yl:yh, xl:xh].astype(np.float32)
                    if self.correct_lam:
                        lam_out[i] -= (yh - yl) * (xh - xl) / (tensor.shape[-2] * tensor.shape[-1])
                    else:
                        lam_out[i] = lam
            else:
                mixed = batch[i][0].astype(np.float32) * lam + \
                        batch[batch_size - i - 1][0].astype(np.float32) * (1 - lam)
                lam_out[i] = lam
            np.round(mixed, out=mixed)
            tensor[i] += torch.from_numpy(mixed.astype(np.uint8))
        return lam_out


class FastCollateMixupBatchwise(FastCollateMixup):
    """Fast Collate Mixup that applies same params to whole batch

    NOTE this is for experimentation, may remove at some point
    """

    def __init__(self, mixup_alpha=1., label_smoothing=0.1, num_classes=1000, mode=MixupMode.MIXUP):
        super(FastCollateMixupBatchwise, self).__init__(mixup_alpha, label_smoothing, num_classes, mode)

    def _do_mix(self, tensor, batch):
        batch_size = len(batch)
        lam_out = torch.ones(batch_size)
        lam = 1.
        cutmix = _resolve_mode(self.mode) == MixupMode.CUTMIX
        if self.mixup_enabled:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            if cutmix and self.correct_lam:
                ratio = math.sqrt(1. - lam)
                yl, yh, xl, xh = rand_bbox(batch[0][0].shape, ratio)
                lam = 1 - (yh - yl) * (xh - xl) / (tensor.shape[-2] * tensor.shape[-1])

        for i in range(batch_size):
            if cutmix:
                mixed = batch[i][0].astype(np.float32)
                if lam != 1:
                    mixed[:, yl:yh, xl:xh] = batch[batch_size - i - 1][0][:, yl:yh, xl:xh].astype(np.float32)
                    lam_out[i] -= (yh - yl) * (xh - xl) / (tensor.shape[-2] * tensor.shape[-1])
            else:
                mixed = batch[i][0].astype(np.float32) * lam + \
                        batch[batch_size - i - 1][0].astype(np.float32) * (1 - lam)
            np.round(mixed, out=mixed)
            tensor[i] += torch.from_numpy(mixed.astype(np.uint8))
        return lam
