from __future__ import absolute_import

import random
import math
import numpy as np
import torch


class RandomErasingNumpy:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This 'Numpy' variant of RandomErasing is intended to be applied on a per
        image basis after transforming the image to uint8 numpy array in
        range 0-255 prior to tensor conversion and normalization
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(
            self,
            probability=0.5, sl=0.02, sh=1/3, min_aspect=0.3,
            per_pixel=False, rand_color=False,
            pl=0, ph=255, mean=[255 * 0.485, 255 * 0.456, 255 * 0.406],
            out_type=np.uint8):
        self.probability = probability
        if not per_pixel and not rand_color:
            self.mean = np.array(mean).round().astype(out_type)
        else:
            self.mean = None
        self.sl = sl
        self.sh = sh
        self.min_aspect = min_aspect
        self.pl = pl
        self.ph = ph
        self.per_pixel = per_pixel  # per pixel random, bounded by [pl, ph]
        self.rand_color = rand_color  # per block random, bounded by [pl, ph]
        self.out_type = out_type

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        chan, img_h, img_w = img.shape
        area = img_h * img_w
        for attempt in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.min_aspect, 1 / self.min_aspect)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if self.rand_color:
                c = np.random.randint(self.pl, self.ph + 1, (chan,), self.out_type)
            elif not self.per_pixel:
                c = self.mean[:chan]
            if w < img_w and h < img_h:
                top = random.randint(0, img_h - h)
                left = random.randint(0, img_w - w)
                if self.per_pixel:
                    img[:, top:top + h, left:left + w] = np.random.randint(
                        self.pl, self.ph + 1, (chan, h, w), self.out_type)
                else:
                    img[:, top:top + h, left:left + w] = c
                return img

        return img


class RandomErasingTorch:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This 'Torch' variant of RandomErasing is intended to be applied to a full batch
        tensor after it has been normalized by dataset mean and std.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
    """

    def __init__(
            self,
            probability=0.5, sl=0.02, sh=1/3, min_aspect=0.3,
            per_pixel=False, rand_color=False):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.min_aspect = min_aspect
        self.per_pixel = per_pixel  # per pixel random, bounded by [pl, ph]
        self.rand_color = rand_color  # per block random, bounded by [pl, ph]

    def __call__(self, batch):
        batch_size, chan, img_h, img_w = batch.size()
        area = img_h * img_w
        for i in range(batch_size):
            if random.random() > self.probability:
                continue
            img = batch[i]
            for attempt in range(100):
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.min_aspect, 1 / self.min_aspect)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if self.rand_color:
                    c = torch.empty((chan, 1, 1), dtype=batch.dtype).normal_().cuda()
                elif not self.per_pixel:
                    c = torch.zeros((chan, 1, 1), dtype=batch.dtype).cuda()
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    if self.per_pixel:
                        img[:, top:top + h, left:left + w] = torch.empty(
                            (chan, h, w), dtype=batch.dtype).normal_().cuda()
                    else:
                        img[:, top:top + h, left:left + w] = c
                    break

        return batch
