""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by / Copyright 2020 Ross Wightman
"""

import numpy as np
import torch
import math
import numbers


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


def rand_bbox(size, lam, border=0., count=None):
    ratio = math.sqrt(1 - lam)
    img_h, img_w = size[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(border * cut_h), int(border * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(size, minmax, count=None):
    assert len(minmax) == 2
    img_h, img_w = size[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


def cutmix_batch(input, target, alpha=0.2, num_classes=1000, smoothing=0.1, disable=False, correct_lam=False):
    lam = 1.
    if not disable:
        lam = np.random.beta(alpha, alpha)
    if lam != 1:
        yl, yh, xl, xh = rand_bbox(input.size(), lam)
        input[:, :, yl:yh, xl:xh] = input.flip(0)[:, :, yl:yh, xl:xh]
        if correct_lam:
            lam = 1. - (yh - yl) * (xh - xl) / float(input.shape[-2] * input.shape[-1])
    target = mixup_target(target, num_classes, lam, smoothing)
    return input, target


def mix_batch(
        input, target, mixup_alpha=0.2, cutmix_alpha=0., prob=1.0, switch_prob=.5,
        num_classes=1000, smoothing=0.1, disable=False):
    # FIXME test this version
    if np.random.rand() > prob:
        return input, target
    use_cutmix = cutmix_alpha > 0. and np.random.rand() <= switch_prob
    if use_cutmix:
        return cutmix_batch(input, target, cutmix_alpha, num_classes, smoothing, disable)
    else:
        return mixup_batch(input, target, mixup_alpha, num_classes, smoothing, disable)


class FastCollateMixup:
    """Fast Collate Mixup/Cutmix that applies different params to each element or whole batch

    NOTE once experiments are done, one of the three variants will remain with this class name

    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 elementwise=False, correct_lam=True, label_smoothing=0.1, num_classes=1000):
        """

        Args:
            mixup_alpha (float): mixup alpha value, mixup is active if > 0.
            cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
            cutmix_minmax (float): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None
            prob (float): probability of applying mixup or cutmix per batch or element
            switch_prob (float): probability of using cutmix instead of mixup when both active
            elementwise (bool): apply mixup/cutmix params per batch element instead of per batch
            label_smoothing (float):
            num_classes (int):
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.elementwise = elementwise
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _mix_elem(self, output, batch):
        batch_size = len(batch)
        lam_out = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size).astype(np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size).astype(np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam_out = np.where(np.random.rand(batch_size) < self.prob, lam_mix.astype(np.float32), lam_out)

        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_out[i]
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix[i]:
                    mixed = mixed.copy()
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                    lam_out[i] = lam
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    lam_out[i] = lam
                    np.round(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        return torch.tensor(lam_out).unsqueeze(1)

    def _mix_batch(self, output, batch):
        batch_size = len(batch)
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)

        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)

        for i in range(batch_size):
            j = batch_size - i - 1
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy()
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.round(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        return lam

    def __call__(self, batch):
        batch_size = len(batch)
        assert batch_size % 2 == 0, 'Batch size should be even when using this'
        output = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        if self.elementwise:
            lam = self._mix_elem(output, batch)
        else:
            lam = self._mix_batch(output, batch)
        target = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device='cpu')

        return output, target

