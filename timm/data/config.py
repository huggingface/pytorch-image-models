import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Union

from .constants import *


_logger = logging.getLogger(__name__)


@dataclass
class AugCfg:
    scale_range: Tuple[float, float] = (0.08, 1.0)
    ratio_range: Tuple[float, float] = (3 / 4, 4 / 3)
    hflip_prob: float = 0.5
    vflip_prob: float = 0.

    color_jitter: float = 0.4
    auto_augment: Optional[str] = None

    re_prob: float = 0.
    re_mode: str = 'const'
    re_count: int = 1

    num_aug_splits: int = 0


@dataclass
class PreprocessCfg:
    input_size: Tuple[int, int, int] = (3, 224, 224)
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD
    interpolation: str = 'bilinear'
    crop_pct: float = 0.875
    aug: AugCfg = None


@dataclass
class MixupCfg:
    prob: float = 1.0
    switch_prob: float = 0.5
    mixup_alpha: float = 1.
    cutmix_alpha: float = 0.
    cutmix_minmax: Optional[Tuple[float, float]] = None
    mode: str = 'batch'
    correct_lam: bool = True
    label_smoothing: float = 0.1
    num_classes: int = 0


def resolve_data_config(args, default_cfg={}, model=None, use_test_size=False, verbose=False):
    new_config = {}
    default_cfg = default_cfg
    if not default_cfg and model is not None and hasattr(model, 'default_cfg'):
        default_cfg = model.default_cfg

    # Resolve input/image size
    in_chans = 3
    if 'chans' in args and args['chans'] is not None:
        in_chans = args['chans']

    input_size = (in_chans, 224, 224)
    if 'input_size' in args and args['input_size'] is not None:
        assert isinstance(args['input_size'], (tuple, list))
        assert len(args['input_size']) == 3
        input_size = tuple(args['input_size'])
        in_chans = input_size[0]  # input_size overrides in_chans
    elif 'img_size' in args and args['img_size'] is not None:
        assert isinstance(args['img_size'], int)
        input_size = (in_chans, args['img_size'], args['img_size'])
    else:
        if use_test_size and 'test_input_size' in default_cfg:
            input_size = default_cfg['test_input_size']
        elif 'input_size' in default_cfg:
            input_size = default_cfg['input_size']
    new_config['input_size'] = input_size

    # resolve interpolation method
    new_config['interpolation'] = 'bicubic'
    if 'interpolation' in args and args['interpolation']:
        new_config['interpolation'] = args['interpolation']
    elif 'interpolation' in default_cfg:
        new_config['interpolation'] = default_cfg['interpolation']

    # resolve dataset + model mean for normalization
    new_config['mean'] = IMAGENET_DEFAULT_MEAN
    if 'mean' in args and args['mean'] is not None:
        mean = tuple(args['mean'])
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        new_config['mean'] = mean
    elif 'mean' in default_cfg:
        new_config['mean'] = default_cfg['mean']

    # resolve dataset + model std deviation for normalization
    new_config['std'] = IMAGENET_DEFAULT_STD
    if 'std' in args and args['std'] is not None:
        std = tuple(args['std'])
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        new_config['std'] = std
    elif 'std' in default_cfg:
        new_config['std'] = default_cfg['std']

    # resolve default crop percentage
    crop_pct = DEFAULT_CROP_PCT
    if 'crop_pct' in args and args['crop_pct'] is not None:
        crop_pct = args['crop_pct']
    else:
        if use_test_size and 'test_crop_pct' in default_cfg:
            crop_pct = default_cfg['test_crop_pct']
        elif 'crop_pct' in default_cfg:
            crop_pct = default_cfg['crop_pct']
    new_config['crop_pct'] = crop_pct

    if getattr(args, 'mixup', 0) > 0 \
            or getattr(args, 'cutmix', 0) > 0. \
            or getattr(args, 'cutmix_minmax', None) is not None:
        new_config['mixup'] = dict(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.num_classes)

    if verbose:
        _logger.info('Data processing configuration for current model + dataset:')
        for n, v in new_config.items():
            _logger.info('\t%s: %s' % (n, str(v)))

    return new_config
