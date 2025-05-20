""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2019, Ross Wightman
"""
import math
from typing import Optional, Tuple, Union

import torch
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, \
    ResizeKeepRatio, CenterCropOrPad, RandomCropOrPad, TrimBorder, MaybeToTensor, MaybePILToTensor
from timm.data.naflex_transforms import RandomResizedCropToSequence, ResizeToSequence, Patchify
from timm.data.random_erasing import RandomErasing


def transforms_noaug_train(
        img_size: Union[int, Tuple[int, int]] = 224,
        interpolation: str = 'bilinear',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        use_prefetcher: bool = False,
        normalize: bool = True,
):
    """ No-augmentation image transforms for training.

    Args:
        img_size: Target image size.
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalization tensor output w/ provided mean/std (if prefetcher not used).

    Returns:

    """
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        transforms.Resize(img_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size)
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [MaybePILToTensor()]
    elif not normalize:
        # when normalize disabled, converted to tensor without scaling, keep original dtype
        tfl += [MaybePILToTensor()]
    else:
        tfl += [
            MaybeToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std)
            )
        ]
    return transforms.Compose(tfl)


def transforms_imagenet_train(
        img_size: Union[int, Tuple[int, int]] = 224,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
        train_crop_mode: Optional[str] = None,
        hflip: float = 0.5,
        vflip: float = 0.,
        color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        color_jitter_prob: Optional[float] = None,
        force_color_jitter: bool = False,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        interpolation: str = 'random',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_num_splits: int = 0,
        use_prefetcher: bool = False,
        normalize: bool = True,
        separate: bool = False,
        naflex: bool = False,
        patch_size: Union[int, Tuple[int, int]] = 16,
        max_seq_len: int = 576,  # 24x24 for 16x16 patch
        patchify: bool = False,
):
    """ ImageNet-oriented image transforms for training.

    Args:
        img_size: Target image size.
        train_crop_mode: Training random crop mode ('rrc', 'rkrc', 'rkrr').
        scale: Random resize scale range (crop area, < 1.0 => zoom in).
        ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug).
        force_color_jitter: Force color jitter where it is normally disabled (ie with RandAugment on).
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        auto_augment: Auto augment configuration string (see auto_augment.py).
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalize tensor output w/ provided mean/std (if prefetcher not used).
        separate: Output transforms in 3-stage tuple.
        naflex: Enable NaFlex mode, sequence constrained patch output
        patch_size: Patch size for NaFlex mode.
        max_seq_len: Max sequence length for NaFlex mode.

    Returns:
        If separate==True, the transforms are returned as a tuple of 3 separate transforms
          for use in a mixing dataset that passes
            * all data through the first (primary) transform, called the 'clean' data
            * a portion of the data through the secondary transform
            * normalizes and converts the branches above with the third, final transform
    """
    train_crop_mode = train_crop_mode or 'rrc'
    assert train_crop_mode in {'rrc', 'rkrc', 'rkrr'}

    primary_tfl = []
    if naflex:
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3. / 4., 4. / 3.))  # default imagenet ratio range
        primary_tfl += [RandomResizedCropToSequence(
            patch_size=patch_size,
            max_seq_len=max_seq_len,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation
        )]
    else:
        if train_crop_mode in ('rkrc', 'rkrr'):
            # FIXME integration of RKR is a WIP
            scale = tuple(scale or (0.8, 1.00))
            ratio = tuple(ratio or (0.9, 1/.9))
            primary_tfl += [
                ResizeKeepRatio(
                    img_size,
                    interpolation=interpolation,
                    random_scale_prob=0.5,
                    random_scale_range=scale,
                    random_scale_area=True,  # scale compatible with RRC
                    random_aspect_prob=0.5,
                    random_aspect_range=ratio,
                ),
                CenterCropOrPad(img_size, padding_mode='reflect')
                if train_crop_mode == 'rkrc' else
                RandomCropOrPad(img_size, padding_mode='reflect')
            ]
        else:
            scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
            ratio = tuple(ratio or (3. / 4., 4. / 3.))  # default imagenet ratio range
            primary_tfl += [
                RandomResizedCropAndInterpolation(
                    img_size,
                    scale=scale,
                    ratio=ratio,
                    interpolation=interpolation,
                )
            ]

    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not (force_color_jitter or '3a' in auto_augment)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

    if color_jitter is not None and not disable_color_jitter:
        # color jitter is enabled when not using AA or when forced
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        if color_jitter_prob is not None:
            secondary_tfl += [
                transforms.RandomApply([
                        transforms.ColorJitter(*color_jitter),
                    ],
                    p=color_jitter_prob
                )
            ]
        else:
            secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    if grayscale_prob:
        secondary_tfl += [transforms.RandomGrayscale(p=grayscale_prob)]

    if gaussian_blur_prob:
        secondary_tfl += [
            transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=23),  # hardcoded for now
                ],
                p=gaussian_blur_prob,
            )
        ]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [MaybePILToTensor()]
    elif not normalize:
        # when normalize disable, converted to tensor without scaling, keeps original dtype
        final_tfl += [MaybePILToTensor()]
    else:
        final_tfl += [
            MaybeToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]
        if re_prob > 0.:
            final_tfl += [
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device='cpu',
                )
            ]

    if patchify:
        final_tfl += [Patchify(patch_size=patch_size)]

    if separate:
        return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_imagenet_eval(
        img_size: Union[int, Tuple[int, int]] = 224,
        crop_pct: Optional[float] = None,
        crop_mode: Optional[str] = None,
        crop_border_pixels: Optional[int] = None,
        interpolation: str = 'bilinear',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        use_prefetcher: bool = False,
        normalize: bool = True,
        naflex: bool = False,
        patch_size: Union[int, Tuple[int, int]] = 16,
        max_seq_len: int = 576,  # 24x24 for 16x16 patch
        patchify: bool = False,
):
    """ ImageNet-oriented image transform for evaluation and inference.

    Args:
        img_size: Target image size.
        crop_pct: Crop percentage. Defaults to 0.875 when None.
        crop_mode: Crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
        crop_border_pixels: Trim a border of specified # pixels around edge of original image.
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalize tensor output w/ provided mean/std (if prefetcher not used).
        naflex: Enable NaFlex mode, sequence constrained patch output
        patch_size: Patch size for NaFlex mode.
        max_seq_len: Max sequence length for NaFlex mode.
        patchify: Patchify the output instead of relying on prefetcher

    Returns:
        Composed transform pipeline
    """
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        scale_size = tuple([math.floor(x / crop_pct) for x in img_size])
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = (scale_size, scale_size)

    tfl = []

    if crop_border_pixels:
        tfl += [TrimBorder(crop_border_pixels)]

    if naflex:
        tfl += [ResizeToSequence(
            patch_size=patch_size,
            max_seq_len=max_seq_len,
            interpolation=interpolation,
        )]
    else:
        if crop_mode == 'squash':
            # squash mode scales each edge to 1/pct of target, then crops
            # aspect ratio is not preserved, no img lost if crop_pct == 1.0
            tfl += [
                transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
                transforms.CenterCrop(img_size),
            ]
        elif crop_mode == 'border':
            # scale the longest edge of image to 1/pct of target edge, add borders to pad, then crop
            # no image lost if crop_pct == 1.0
            fill = [round(255 * v) for v in mean]
            tfl += [
                ResizeKeepRatio(scale_size, interpolation=interpolation, longest=1.0),
                CenterCropOrPad(img_size, fill=fill),
            ]
        else:
            # default crop model is center
            # aspect ratio is preserved, crops center within image, no borders are added, image is lost
            if scale_size[0] == scale_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
                tfl += [
                    transforms.Resize(scale_size[0], interpolation=str_to_interp_mode(interpolation))
                ]
            else:
                # resize the shortest edge to matching target dim for non-square target
                tfl += [ResizeKeepRatio(scale_size)]
            tfl += [transforms.CenterCrop(img_size)]

    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [MaybePILToTensor()]
    elif not normalize:
        # when normalize disabled, converted to tensor without scaling, keeps original dtype
        tfl += [MaybePILToTensor()]
    else:
        tfl += [
            MaybeToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]

    if patchify:
        tfl += [Patchify(patch_size=patch_size)]

    return transforms.Compose(tfl)


def create_transform(
        input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 224,
        is_training: bool = False,
        no_aug: bool = False,
        train_crop_mode: Optional[str] = None,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
        hflip: float = 0.5,
        vflip: float = 0.,
        color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        color_jitter_prob: Optional[float] = None,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        interpolation: str = 'bilinear',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_num_splits: int = 0,
        crop_pct: Optional[float] = None,
        crop_mode: Optional[str] = None,
        crop_border_pixels: Optional[int] = None,
        tf_preprocessing: bool = False,
        use_prefetcher: bool = False,
        normalize: bool = True,
        separate: bool = False,
        naflex: bool = False,
        patch_size: Union[int, Tuple[int, int]] = 16,
        max_seq_len: int = 576,  # 24x24 for 16x16 patch
        patchify: bool = False
):
    """

    Args:
        input_size: Target input size (channels, height, width) tuple or size scalar.
        is_training: Return training (random) transforms.
        no_aug: Disable augmentation for training (useful for debug).
        train_crop_mode: Training random crop mode ('rrc', 'rkrc', 'rkrr').
        scale: Random resize scale range (crop area, < 1.0 => zoom in).
        ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug).
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        auto_augment: Auto augment configuration string (see auto_augment.py).
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        crop_pct: Inference crop percentage (output size / resize size).
        crop_mode: Inference crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
        crop_border_pixels: Inference crop border of specified # pixels around edge of original image.
        tf_preprocessing: Use TF 1.0 inference preprocessing for testing model ports
        use_prefetcher: Pre-fetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalization tensor output w/ provided mean/std (if prefetcher not used).
        separate: Output transforms in 3-stage tuple.

    Returns:
        Composed transforms or tuple thereof
    """
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training,
            size=img_size,
            interpolation=interpolation,
        )
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=interpolation,
                mean=mean,
                std=std,
                use_prefetcher=use_prefetcher,
                normalize=normalize,
            )
        elif is_training:
            transform = transforms_imagenet_train(
                img_size,
                train_crop_mode=train_crop_mode,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                color_jitter=color_jitter,
                color_jitter_prob=color_jitter_prob,
                grayscale_prob=grayscale_prob,
                gaussian_blur_prob=gaussian_blur_prob,
                auto_augment=auto_augment,
                interpolation=interpolation,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                use_prefetcher=use_prefetcher,
                normalize=normalize,
                separate=separate,
                naflex=naflex,
                patch_size=patch_size,
                max_seq_len=max_seq_len,
                patchify=patchify,
            )
        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                mean=mean,
                std=std,
                crop_pct=crop_pct,
                crop_mode=crop_mode,
                crop_border_pixels=crop_border_pixels,
                use_prefetcher=use_prefetcher,
                normalize=normalize,
                naflex=naflex,
                patch_size=patch_size,
                max_seq_len=max_seq_len,
                patchify=patchify,
            )

    return transform
