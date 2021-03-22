""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn


from .features import FeatureListNet, FeatureDictNet, FeatureHookNet
from .hub import has_hf_hub, download_cached_file, load_state_dict_from_hf, load_state_dict_from_url
from .layers import Conv2dSame, Linear


_logger = logging.getLogger(__name__)


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_custom_pretrained(model, default_cfg=None, load_fn=None, progress=False, check_hash=False):
    r"""Loads a custom (read non .pth) weight file

    Downloads checkpoint file into cache-dir like torch.hub based loaders, but calls
    a passed in custom load fun, or the `load_pretrained` model member fn.

    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        model: The instantiated model to load weights into
        default_cfg (dict): Default pretrained model cfg
        load_fn: An external stand alone fn that loads weights into provided model, otherwise a fn named
            'laod_pretrained' on the model will be called if it exists
        progress (bool, optional): whether or not to display a progress bar to stderr. Default: False
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file. Default: False
    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)
    if not pretrained_url:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    cached_file = download_cached_file(default_cfg['url'], check_hash=check_hash, progress=progress)

    if load_fn is not None:
        load_fn(model, cached_file)
    elif hasattr(model, 'load_pretrained'):
        model.load_pretrained(cached_file)
    else:
        _logger.warning("Valid function to load pretrained weights is not available, using random initialization.")


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def load_pretrained(model, default_cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, progress=False):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        default_cfg (Optional[Dict]): default configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
        progress (bool): enable progress bar for weight download

    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)
    hf_hub_id = default_cfg.get('hf_hub', None)
    if not pretrained_url and not hf_hub_id:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if hf_hub_id and has_hf_hub(necessary=not pretrained_url):
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({hf_hub_id})')
        state_dict = load_state_dict_from_hf(hf_hub_id)
    else:
        _logger.info(f'Loading pretrained weights from url ({pretrained_url})')
        state_dict = load_state_dict_from_url(pretrained_url, progress=progress, map_location='cpu')
    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = default_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifier_name = default_cfg.get('classifier', None)
    label_offset = default_cfg.get('label_offset', 0)
    if classifier_name is not None:
        if num_classes != default_cfg['num_classes']:
            # completely discard fully connected if model num_classes doesn't match pretrained weights
            del state_dict[classifier_name + '.weight']
            del state_dict[classifier_name + '.bias']
            strict = False
        elif label_offset > 0:
            # special case for pretrained weights with an extra background class in pretrained weights
            classifier_weight = state_dict[classifier_name + '.weight']
            state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
            classifier_bias = state_dict[classifier_name + '.bias']
            state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_string(parent_module, model_string):
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            # FIXME extra checks to ensure this is actually the FC classifier layer and not a diff Linear layer?
            num_features = state_dict[n + '.weight'][1]
            new_fc = Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module


def adapt_model_from_file(parent_module, model_variant):
    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
    with open(adapt_file, 'r') as f:
        return adapt_model_from_string(parent_module, f.read().strip())


def default_cfg_for_features(default_cfg):
    default_cfg = deepcopy(default_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'crop_pct', 'classifier', 'global_pool')  # add default final pool size?
    for tr in to_remove:
        default_cfg.pop(tr, None)
    return default_cfg


def overlay_external_default_cfg(default_cfg, kwargs):
    """ Overlay 'external_default_cfg' in kwargs on top of default_cfg arg.
    """
    external_default_cfg = kwargs.pop('external_default_cfg', None)
    if external_default_cfg:
        default_cfg.pop('url', None)  # url should come from external cfg
        default_cfg.pop('hf_hub', None)  # hf hub id should come from external cfg
        default_cfg.update(external_default_cfg)


def set_default_kwargs(kwargs, names, default_cfg):
    for n in names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # default_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = default_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, default_cfg[n])


def filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    FIXME this sequence of overlay default_cfg, set default kwargs, filter kwargs
    could/should be replaced by an improved configuration mechanism

    Args:
        default_cfg: input default_cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Overlay default cfg values from `external_default_cfg` if it exists in kwargs
    overlay_external_default_cfg(default_cfg, kwargs)
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    set_default_kwargs(kwargs, names=('num_classes', 'global_pool', 'in_chans'), default_cfg=default_cfg)
    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    filter_kwargs(kwargs, names=kwargs_filter)


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        pretrained_custom_load: bool = False,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        default_cfg (dict): model's default pretrained/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
        pretrained_custom_load (bool): use custom load fn, to load numpy or other non PyTorch weights
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}
    default_cfg = deepcopy(default_cfg) if default_cfg else {}
    update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter)
    default_cfg.setdefault('architecture', variant)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    # Build the model
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = default_cfg
    
    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model)
        else:
            load_pretrained(
                model,
                num_classes=num_classes_pretrained,
                in_chans=kwargs.get('in_chans', 3),
                filter_fn=pretrained_filter_fn,
                strict=pretrained_strict)

    # Wrap the model in a feature extraction module if enabled
    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.default_cfg = default_cfg_for_features(default_cfg)  # add back default_cfg
    
    return model


def model_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params based on ordering
        return [p for p in model.parameters()][:-2]
    else:
        return model.parameters()
