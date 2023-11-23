import logging
from .constants import *


_logger = logging.getLogger(__name__)


def resolve_data_config(
        args=None,
        pretrained_cfg=None,
        model=None,
        use_test_size=False,
        verbose=False
):
    assert model or args or pretrained_cfg, "At least one of model, args, or pretrained_cfg required for data config."
    args = args or {}
    pretrained_cfg = pretrained_cfg or {}
    if not pretrained_cfg and model is not None and hasattr(model, 'pretrained_cfg'):
        pretrained_cfg = model.pretrained_cfg
    data_config = {}

    # Resolve input/image size
    in_chans = 3
    if args.get('in_chans', None) is not None:
        in_chans = args['in_chans']
    elif args.get('chans', None) is not None:
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
        if use_test_size and pretrained_cfg.get('test_input_size', None) is not None:
            input_size = pretrained_cfg['test_input_size']
        elif pretrained_cfg.get('input_size', None) is not None:
            input_size = pretrained_cfg['input_size']
    data_config['input_size'] = input_size

    # resolve interpolation method
    data_config['interpolation'] = 'bicubic'
    if args.get('interpolation', None):
        data_config['interpolation'] = args['interpolation']
    elif pretrained_cfg.get('interpolation', None):
        data_config['interpolation'] = pretrained_cfg['interpolation']

    # resolve dataset + model mean for normalization
    data_config['mean'] = IMAGENET_DEFAULT_MEAN
    if args.get('mean', None) is not None:
        mean = tuple(args['mean'])
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        data_config['mean'] = mean
    elif pretrained_cfg.get('mean', None):
        data_config['mean'] = pretrained_cfg['mean']

    # resolve dataset + model std deviation for normalization
    data_config['std'] = IMAGENET_DEFAULT_STD
    if args.get('std', None) is not None:
        std = tuple(args['std'])
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        data_config['std'] = std
    elif pretrained_cfg.get('std', None):
        data_config['std'] = pretrained_cfg['std']

    # resolve default inference crop
    crop_pct = DEFAULT_CROP_PCT
    if args.get('crop_pct', None):
        crop_pct = args['crop_pct']
    else:
        if use_test_size and pretrained_cfg.get('test_crop_pct', None):
            crop_pct = pretrained_cfg['test_crop_pct']
        elif pretrained_cfg.get('crop_pct', None):
            crop_pct = pretrained_cfg['crop_pct']
    data_config['crop_pct'] = crop_pct

    # resolve default crop percentage
    crop_mode = DEFAULT_CROP_MODE
    if args.get('crop_mode', None):
        crop_mode = args['crop_mode']
    elif pretrained_cfg.get('crop_mode', None):
        crop_mode = pretrained_cfg['crop_mode']
    data_config['crop_mode'] = crop_mode

    if verbose:
        _logger.info('Data processing configuration for current model + dataset:')
        for n, v in data_config.items():
            _logger.info('\t%s: %s' % (n, str(v)))

    return data_config


def resolve_model_data_config(
        model,
        args=None,
        pretrained_cfg=None,
        use_test_size=False,
        verbose=False,
):
    """ Resolve Model Data Config
    This is equivalent to resolve_data_config() but with arguments re-ordered to put model first.

    Args:
        model (nn.Module): the model instance
        args (dict): command line arguments / configuration in dict form (overrides pretrained_cfg)
        pretrained_cfg (dict): pretrained model config (overrides pretrained_cfg attached to model)
        use_test_size (bool): use the test time input resolution (if one exists) instead of default train resolution
        verbose (bool): enable extra logging of resolved values

    Returns:
        dictionary of config
    """
    return resolve_data_config(
        args=args,
        pretrained_cfg=pretrained_cfg,
        model=model,
        use_test_size=use_test_size,
        verbose=verbose,
    )
