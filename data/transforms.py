import torch
from torchvision import transforms
from PIL import Image
import math
import numpy as np
from data.random_erasing import RandomErasingNumpy

DEFAULT_CROP_PCT = 0.875

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)


def resolve_data_config(model, args, default_cfg={}, verbose=True):
    new_config = {}
    default_cfg = default_cfg
    if not default_cfg and hasattr(model, 'default_cfg'):
        default_cfg = model.default_cfg

    # Resolve input/image size
    # FIXME grayscale/chans arg to use different # channels?
    in_chans = 3
    input_size = (in_chans, 224, 224)
    if args.img_size is not None:
        # FIXME support passing img_size as tuple, non-square
        assert isinstance(args.img_size, int)
        input_size = (in_chans, args.img_size, args.img_size)
    elif 'input_size' in default_cfg:
        input_size = default_cfg['input_size']
    new_config['input_size'] = input_size

    # resolve interpolation method
    new_config['interpolation'] = 'bilinear'
    if args.interpolation:
        new_config['interpolation'] = args.interpolation
    elif 'interpolation' in default_cfg:
        new_config['interpolation'] = default_cfg['interpolation']

    # resolve dataset + model mean for normalization
    new_config['mean'] = get_mean_by_model(args.model)
    if args.mean is not None:
        mean = tuple(args.mean)
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        new_config['mean'] = mean
    elif 'mean' in default_cfg:
        new_config['mean'] = default_cfg['mean']

    # resolve dataset + model std deviation for normalization
    new_config['std'] = get_std_by_model(args.model)
    if args.std is not None:
        std = tuple(args.std)
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        new_config['std'] = std
    elif 'std' in default_cfg:
        new_config['std'] = default_cfg['std']

    # resolve default crop percentage
    new_config['crop_pct'] = DEFAULT_CROP_PCT
    if 'crop_pct' in default_cfg:
        new_config['crop_pct'] = default_cfg['crop_pct']

    if verbose:
        print('Data processing configuration for current model + dataset:')
        for n, v in new_config.items():
            print('\t%s: %s' % (n, str(v)))

    return new_config


def get_mean_by_name(name):
    if name == 'dpn':
        return IMAGENET_DPN_MEAN
    elif name == 'inception' or name == 'le':
        return IMAGENET_INCEPTION_MEAN
    else:
        return IMAGENET_DEFAULT_MEAN


def get_std_by_name(name):
    if name == 'dpn':
        return IMAGENET_DPN_STD
    elif name == 'inception' or name == 'le':
        return IMAGENET_INCEPTION_STD
    else:
        return IMAGENET_DEFAULT_STD


def get_mean_by_model(model_name):
    model_name = model_name.lower()
    if 'dpn' in model_name:
        return IMAGENET_DPN_STD
    elif 'ception' in model_name or 'nasnet' in model_name:
        return IMAGENET_INCEPTION_MEAN
    else:
        return IMAGENET_DEFAULT_MEAN


def get_std_by_model(model_name):
    model_name = model_name.lower()
    if 'dpn' in model_name:
        return IMAGENET_DEFAULT_STD
    elif 'ception' in model_name or 'nasnet' in model_name:
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


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def transforms_imagenet_train(
        img_size=224,
        scale=(0.1, 1.0),
        color_jitter=(0.4, 0.4, 0.4),
        interpolation='bilinear',
        random_erasing=0.4,
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD
):

    tfl = [
        transforms.RandomResizedCrop(
            img_size, scale=scale,
            interpolation=_pil_interp(interpolation)),
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
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if random_erasing > 0.:
            tfl.append(RandomErasingNumpy(random_erasing, per_pixel=True))
    return transforms.Compose(tfl)


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, tuple):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, _pil_interp(interpolation)),
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

    return transforms.Compose(tfl)
