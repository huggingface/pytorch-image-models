from data.constants import *


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
