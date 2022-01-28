""" Random Erasing (Cutout)

Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2019, Ross Wightman
"""
import random
import math
import torch


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
         count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', count=1, num_splits=0):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.count = count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'

    def _erase(self, img, chan, img_h, img_w, dtype):
        device = img.device
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = random.randint(1, self.count) if self.count > 1 else self.count
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w), dtype=dtype, device=device)
                    break

    def __call__(self, x):
        if len(x.size()) == 3:
            self._erase(x, *x.shape, x.dtype)
        else:
            batch_size, chan, img_h, img_w = x.shape
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(x[i], chan, img_h, img_w, x.dtype)
        return x


class RandomErasingMasked:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed for each box (count)
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is between 0 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', count=1, num_splits=0):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.mode = mode   # FIXME currently ignored, add back options besides normal mean=0, std=1 noise?
        self.count = count
        self.num_splits = num_splits

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size, _, img_h, img_w = x.shape
        batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0

        # NOTE simplified from v1 with with one count value and same prob applied for all
        enable = (torch.empty((batch_size, self.count), device=device).uniform_() < self.probability).float()
        enable = enable / enable.sum(dim=1, keepdim=True).clamp(min=1)
        target_area = torch.empty(
            (batch_size, self.count), device=device).uniform_(self.min_area, self.max_area) * enable
        aspect_ratio = torch.empty((batch_size, self.count), device=device).uniform_(*self.log_aspect_ratio).exp()
        h_coord = torch.arange(0, img_h, device=device).unsqueeze(-1).expand(-1, self.count).float()
        w_coord = torch.arange(0, img_w, device=device).unsqueeze(-1).expand(-1, self.count).float()
        h_mid = torch.rand((batch_size, self.count), device=device) * img_h
        w_mid = torch.rand((batch_size, self.count), device=device) * img_w
        noise = torch.empty_like(x[0]).normal_()

        for i in range(batch_start, batch_size):
            h_half = (img_h / 2) * torch.sqrt(target_area[i] * aspect_ratio[i])  # 1/2 box h
            h_mask = (h_coord > (h_mid[i] - h_half)) & (h_coord < (h_mid[i] + h_half))
            w_half = (img_w / 2) * torch.sqrt(target_area[i] / aspect_ratio[i])  # 1/2 box w
            w_mask = (w_coord > (w_mid[i] - w_half)) & (w_coord < (w_mid[i] + w_half))
            #mask = (h_mask.unsqueeze(1) & w_mask.unsqueeze(0)).any(dim=-1)
            #x[i].copy_(torch.where(mask, noise, x[i]))
            mask = ~(h_mask.unsqueeze(1) & w_mask.unsqueeze(0)).any(dim=-1)
            x[i] = x[i].where(mask, noise)
            #x[i].masked_scatter_(mask, noise)
        return x
