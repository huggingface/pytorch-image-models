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


# FIXME replace these mean/std fn with model factory based values from config dict
def get_model_meanstd(model_name):
    model_name = model_name.lower()
    if 'dpn' in model_name:
        return IMAGENET_DPN_MEAN, IMAGENET_DPN_STD
    elif 'ception' in model_name:
        return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    else:
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_model_mean(model_name):
    model_name = model_name.lower()
    if 'dpn' in model_name:
        return IMAGENET_DPN_STD
    elif 'ception' in model_name:
        return IMAGENET_INCEPTION_MEAN
    else:
        return IMAGENET_DEFAULT_MEAN


def get_model_std(model_name):
    model_name = model_name.lower()
    if 'dpn' in model_name:
        return IMAGENET_DEFAULT_STD
    elif 'ception' in model_name:
        return IMAGENET_INCEPTION_STD
    else:
        return IMAGENET_DEFAULT_STD


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype)


def transforms_imagenet_train(
        img_size=224,
        scale=(0.1, 1.0),
        color_jitter=(0.4, 0.4, 0.4),
        random_erasing=0.4,
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD
):

    tfl = [
        transforms.RandomResizedCrop(
            img_size, scale=scale, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(*color_jitter),
    ]

    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean) * 255,
                std=torch.tensor(std) * 255)
        ]
        if random_erasing > 0.:
            tfl.append(RandomErasingNumpy(random_erasing, per_pixel=True))
    return transforms.Compose(tfl)


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]
        # tfl += [
        #     ToTensor(),
        #     transforms.Normalize(
        #         mean=torch.tensor(mean) * 255,
        #         std=torch.tensor(std) * 255)
        # ]

    return transforms.Compose(tfl)
