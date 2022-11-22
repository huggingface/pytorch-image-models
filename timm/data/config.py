import logging
from .constants import *


_logger = logging.getLogger(__name__)


def resolve_data_config(
        args,
        default_cfg=None,
        model=None,
        use_test_size=False,
        verbose=False
):
    new_config = {}
    default_cfg = default_cfg or {}
    if not default_cfg and model is not None and hasattr(model, 'default_cfg'):
        default_cfg = model.default_cfg

    # Resolve input/image size
    in_chans = 3
    if args.get('chans', None) is not None:
        in_chans = args['chans']

    input_size = (in_chans, 224, 224)
    if args.get('input_size', None) is not None:
        assert isinstance(args['input_size'], (tuple, list))
        assert len(args['input_size']) == 3
        input_size = tuple(args['input_size'])
        in_chans = input_size[0]  # input_size overrides in_chans
    elif args.get('img_size', None) is not None:
        assert isinstance(args['img_size'], int)
        input_size = (in_chans, args['img_size'], args['img_size'])
    else:
        if use_test_size and default_cfg.get('test_input_size', None) is not None:
            input_size = default_cfg['test_input_size']
        elif default_cfg.get('input_size', None) is not None:
            input_size = default_cfg['input_size']
    new_config['input_size'] = input_size

    # resolve interpolation method
    new_config['interpolation'] = 'bicubic'
    if args.get('interpolation', None):
        new_config['interpolation'] = args['interpolation']
    elif default_cfg.get('interpolation', None):
        new_config['interpolation'] = default_cfg['interpolation']

    # resolve dataset + model mean for normalization
    new_config['mean'] = IMAGENET_DEFAULT_MEAN
    if args.get('mean', None) is not None:
        mean = tuple(args['mean'])
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        new_config['mean'] = mean
    elif default_cfg.get('mean', None):
        new_config['mean'] = default_cfg['mean']

    # resolve dataset + model std deviation for normalization
    new_config['std'] = IMAGENET_DEFAULT_STD
    if args.get('std', None) is not None:
        std = tuple(args['std'])
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        new_config['std'] = std
    elif default_cfg.get('std', None):
        new_config['std'] = default_cfg['std']

    # resolve default inference crop
    crop_pct = DEFAULT_CROP_PCT
    if args.get('crop_pct', None):
        crop_pct = args['crop_pct']
    else:
        if use_test_size and default_cfg.get('test_crop_pct', None):
            crop_pct = default_cfg['test_crop_pct']
        elif default_cfg.get('crop_pct', None):
            crop_pct = default_cfg['crop_pct']
    new_config['crop_pct'] = crop_pct

    # resolve default crop percentage
    crop_mode = DEFAULT_CROP_MODE
    if args.get('crop_mode', None):
        crop_mode = args['crop_mode']
    elif default_cfg.get('crop_mode', None):
        crop_mode = default_cfg['crop_mode']
    new_config['crop_mode'] = crop_mode

    if verbose:
        _logger.info('Data processing configuration for current model + dataset:')
        for n, v in new_config.items():
            _logger.info('\t%s: %s' % (n, str(v)))

    return new_config
