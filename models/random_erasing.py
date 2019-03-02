from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
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
            per_pixel=False, random=False,
            pl=0, ph=1., mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = torch.tensor(mean)
        self.sl = sl
        self.sh = sh
        self.min_aspect = min_aspect
        self.pl = pl
        self.ph = ph
        self.per_pixel = per_pixel  # per pixel random, bounded by [pl, ph]
        self.random = random  # per block random, bounded by [pl, ph]

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        chan, img_h, img_w = img.size()
        area = img_h * img_w
        for attempt in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.min_aspect, 1 / self.min_aspect)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            c = torch.empty((chan)).uniform_(self.pl, self.ph) if self.random else self.mean[:chan]
            if w < img_w and h < img_h:
                top = random.randint(0, img_h - h)
                left = random.randint(0, img_w - w)
                if self.per_pixel:
                    img[:, top:top + h, left:left + w] = torch.empty((chan, h, w)).uniform_(self.pl, self.ph)
                else:
                    img[:, top:top + h, left:left + w] = c
                return img

        return img
