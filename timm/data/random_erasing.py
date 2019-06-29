import random
import math
import torch


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    # will revert back to doing normal_() on GPU when it's in next release
    if per_pixel:
        return torch.empty(
            patch_size, dtype=dtype).normal_().to(device=device)
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype).normal_().to(device=device)
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-cannel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
    """

    def __init__(
            self,
            probability=0.5, sl=0.02, sh=1/3, min_aspect=0.3,
            mode='const', device='cuda'):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.min_aspect = min_aspect
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        for attempt in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.min_aspect, 1 / self.min_aspect)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img_w and h < img_h:
                top = random.randint(0, img_h - h)
                left = random.randint(0, img_w - w)
                img[:, top:top + h, left:left + w] = _get_pixels(
                    self.per_pixel, self.rand_color, (chan, h, w),
                    dtype=dtype, device=self.device)
                break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            for i in range(batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input
