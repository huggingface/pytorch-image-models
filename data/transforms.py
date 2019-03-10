import torch
from torchvision import transforms
from PIL import Image
import math
import numpy as np
from data.random_erasing import RandomErasingNumpy

DEFAULT_CROP_PCT = 0.875

IMAGENET_DPN_MEAN = [124 / 255, 117 / 255, 104 / 255]
IMAGENET_DPN_STD = [1 / (.0167 * 255)] * 3
IMAGENET_INCEPTION_MEAN = [0.5, 0.5, 0.5]
IMAGENET_INCEPTION_STD = [0.5, 0.5, 0.5]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class AsNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


def transforms_imagenet_train(
        img_size=224,
        scale=(0.1, 1.0),
        color_jitter=(0.4, 0.4, 0.4),
        random_erasing=0.4):

    tfl = [
        transforms.RandomResizedCrop(img_size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(*color_jitter),
        AsNumpy(),
    ]
    #if random_erasing > 0.:
    #    tfl.append(RandomErasingNumpy(random_erasing, per_pixel=True))
    return transforms.Compose(tfl)


def transforms_imagenet_eval(img_size=224, crop_pct=None):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    scale_size = int(math.floor(img_size / crop_pct))

    return transforms.Compose([
        transforms.Resize(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        AsNumpy(),
    ])
