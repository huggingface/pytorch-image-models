""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2019, Ross Wightman
"""
import math
from typing import Union, Tuple

import torch
from torchvision import transforms

from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.config import PreprocessCfg, AugCfg
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensorNormalize


def transforms_noaug_train(
        img_size: Union[int, Tuple[int]] = 224,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        normalize=False,
        compose=True,
):
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        transforms.Resize(img_size, transforms.InterpolationMode(interpolation)),
        transforms.CenterCrop(img_size)
    ]
    if normalize:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ]
    else:
        # (pre)fetcher and collate will handle tensor conversion and normalize
        tfl += [ToNumpy()]
    return transforms.Compose(tfl) if compose else tfl


def transforms_imagenet_train(
        img_size: Union[int, Tuple[int]] = 224,
        interpolation='random',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        aug_cfg=AugCfg(),
        normalize=False,
        separate=False,
        compose=True,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale_range = tuple(aug_cfg.scale_range or (0.08, 1.0))  # default imagenet scale range
    ratio_range = tuple(aug_cfg.ratio_range or (3. / 4., 4. / 3.))  # default imagenet ratio range

    # 'primary' train transforms include random resize + crop w/ optional horizontal and vertical flipping aug.
    # This is the core of standard ImageNet ResNet and Inception pre-processing
    primary_tfl = [
        RandomResizedCropAndInterpolation(img_size, scale=scale_range, ratio=ratio_range, interpolation=interpolation)]
    if aug_cfg.hflip_prob > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=aug_cfg.hflip_prob)]
    if aug_cfg.vflip_prob > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=aug_cfg.vflip_prob)]

    # 'secondary' transform stage includes either color jitter (could add lighting too) or auto-augmentations
    # such as AutoAugment, RandAugment, AugMix, etc
    secondary_tfl = []
    if aug_cfg.auto_augment:
        aa = aug_cfg.auto_augment
        assert isinstance(aa, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = interpolation
        if aa.startswith('rand'):
            secondary_tfl += [rand_augment_transform(aa, aa_params)]
        elif aa.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(aa, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(aa, aa_params)]
    elif aug_cfg.color_jitter is not None:
        # color jitter is enabled when not using AA
        cj = aug_cfg.color_jitter
        if isinstance(cj, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(cj) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            cj = (float(cj),) * 3
        secondary_tfl += [transforms.ColorJitter(*cj)]

    # 'final' transform stage includes normalization, followed by optional random erasing and tensor conversion
    final_tfl = []
    if normalize:
        final_tfl += [
            ToTensorNormalize(mean=mean, std=std)
        ]
        if aug_cfg.re_prob > 0.:
            final_tfl.append(RandomErasing(
                aug_cfg.re_prob,
                mode=aug_cfg.re_mode,
                count=aug_cfg.re_count,
                num_splits=aug_cfg.num_aug_splits))
    else:
        # when normalize disabled, (pre)fetcher and collate will handle tensor conversion and normalize
        final_tfl += [ToNumpy()]

    if separate:
        # return each transform stage separately
        if compose:
            return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
        else:
            return primary_tfl, secondary_tfl, final_tfl
    else:
        tfl = primary_tfl + secondary_tfl + final_tfl
        return transforms.Compose(tfl) if compose else tfl


def transforms_imagenet_eval(
        img_size: Union[int, Tuple[int]] = 224,
        crop_pct=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        normalize=False,
        compose=True,
):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # FIXME handle case where img is square and we want non aspect preserving resize
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, transforms.InterpolationMode(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if normalize:
        tfl += [
            ToTensorNormalize(mean=mean, std=std)
        ]
    else:
        # (pre)fetcher and collate will handle tensor conversion and normalize
        tfl += [ToNumpy()]

    return transforms.Compose(tfl) if compose else tfl


def create_transform_v2(
        cfg=PreprocessCfg(),
        is_training=False,
        normalize=False,
        separate=False,
        compose=True,
        tf_preprocessing=False,
):
    """
    
    Args:
        cfg: Pre-processing configuration
        is_training (bool): Create transform for training pre-processing
        normalize (bool): Enable normalization in transforms (otherwise handled by fetcher/pre-fetcher)
        separate (bool): Return transforms separated into stages (for train)
        compose (bool): Wrap transforms in transform.Compose(), returns list otherwise
        tf_preprocessing (bool): Use Tensorflow pre-processing (for validation)
    Returns:

    """
    input_size = cfg.input_size
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing:
        assert not normalize, "Expecting normalization to be handled in (pre)fetcher w/ TF preprocessing"
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=cfg.interpolation)
    else:
        if is_training and cfg.aug is None:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=cfg.interpolation,
                normalize=normalize,
                mean=cfg.mean,
                std=cfg.std,
                compose=compose,
            )
        elif is_training:
            transform = transforms_imagenet_train(
                img_size,
                interpolation=cfg.interpolation,
                mean=cfg.mean,
                std=cfg.std,
                aug_cfg=cfg.aug,
                normalize=normalize,
                separate=separate,
                compose=compose,
            )
        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=cfg.interpolation,
                crop_pct=cfg.crop_pct,
                mean=cfg.mean,
                std=cfg.std,
                normalize=normalize,
                compose=compose,
            )

    return transform


def create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    normalize_in_transform = not use_prefetcher
    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation)
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=interpolation,
                mean=mean,
                std=std,
                normalize=normalize_in_transform,
            )
        elif is_training:
            aug_cfg = AugCfg(
                scale_range=scale,
                ratio_range=ratio,
                hflip_prob=hflip,
                vflip_prob=vflip,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                num_aug_splits=re_num_splits,
            )
            transform = transforms_imagenet_train(
                img_size,
                interpolation=interpolation,
                mean=mean,
                std=std,
                aug_cfg=aug_cfg,
                normalize=normalize_in_transform,
                separate=separate
            )
        else:
            assert not separate, "Separate transforms not supported for validation pre-processing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                mean=mean,
                std=std,
                crop_pct=crop_pct,
                normalize=normalize_in_transform,
            )

    return transform
