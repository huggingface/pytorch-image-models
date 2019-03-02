import torch
from torchvision import transforms
from PIL import Image
import math
from models.random_erasing import RandomErasing

DEFAULT_CROP_PCT = 0.875

IMAGENET_DPN_MEAN = [124 / 255, 117 / 255, 104 / 255]
IMAGENET_DPN_STD = [1 / (.0167 * 255)] * 3
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def transforms_imagenet_train(
        model_name,
        img_size=224,
        scale=(0.1, 1.0),
        color_jitter=(0.4, 0.4, 0.4),
        random_erasing=0.4):
    if 'dpn' in model_name:
        normalize = transforms.Normalize(
            mean=IMAGENET_DPN_MEAN,
            std=IMAGENET_DPN_STD)
    elif 'inception' in model_name:
        normalize = LeNormalize()
    else:
        normalize = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD)

    tfl = [
        transforms.RandomResizedCrop(img_size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(*color_jitter),
        transforms.ToTensor()]
    if random_erasing > 0.:
        tfl.append(RandomErasing(random_erasing, per_pixel=True))
    return transforms.Compose(tfl + [normalize])


def transforms_imagenet_eval(model_name, img_size=224, crop_pct=None):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    if 'dpn' in model_name:
        if crop_pct is None:
            # Use default 87.5% crop for model's native img_size
            # but use 100% crop for larger than native as it
            # improves test time results across all models.
            if img_size == 224:
                scale_size = int(math.floor(img_size / DEFAULT_CROP_PCT))
            else:
                scale_size = img_size
        else:
            scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=IMAGENET_DPN_MEAN,
            std=IMAGENET_DPN_STD)
    elif 'inception' in model_name:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = LeNormalize()
    else:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD)

    return transforms.Compose([
        transforms.Resize(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize])
