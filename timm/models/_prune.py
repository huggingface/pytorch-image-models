import logging
import os
import pkgutil
from copy import deepcopy

import torch
from torch import nn as nn

from timm.layers import Conv2dSame, BatchNormAct2d, Linear

__all__ = ['extract_layer', 'set_layer', 'adapt_model_from_string', 'adapt_model_from_file']

_logger = logging.getLogger(__name__)


def extract_layer(model, layer):
    """Extract a layer from a model using dot-separated path.

    Args:
        model: PyTorch model.
        layer: Dot-separated layer path (e.g., 'layer1.0.conv1').

    Returns:
        Extracted module.
    """
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
    """Set a layer in a model using dot-separated path.

    Args:
        model: PyTorch model.
        layer: Dot-separated layer path.
        val: New value for the layer.
    """
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
    """Adapt a model to pruned structure from string specification.

    Args:
        parent_module: Original model to adapt.
        model_string: String containing layer shapes for pruned model.

    Returns:
        Adapted model with pruned layer dimensions.
    """
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    # Extract device and dtype from the parent module
    device = next(parent_module.parameters()).device
    dtype = next(parent_module.parameters()).dtype
    dd = {'device': device, 'dtype': dtype}

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
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None,
                padding=old_module.padding,
                dilation=old_module.dilation,
                groups=g,
                stride=old_module.stride,
                **dd,
            )
            set_layer(new_module, n, new_conv)
        elif isinstance(old_module, BatchNormAct2d):
            new_bn = BatchNormAct2d(
                state_dict[n + '.weight'][0],
                eps=old_module.eps,
                momentum=old_module.momentum,
                affine=old_module.affine,
                track_running_stats=True,
                **dd,
            )
            new_bn.drop = old_module.drop
            new_bn.act = old_module.act
            set_layer(new_module, n, new_bn)
        elif isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0],
                eps=old_module.eps,
                momentum=old_module.momentum,
                affine=old_module.affine,
                track_running_stats=True,
                **dd,
            )
            set_layer(new_module, n, new_bn)
        elif isinstance(old_module, nn.Linear):
            # FIXME extra checks to ensure this is actually the FC classifier layer and not a diff Linear layer?
            num_features = state_dict[n + '.weight'][1]
            new_fc = Linear(
                in_features=num_features,
                out_features=old_module.out_features,
                bias=old_module.bias is not None,
                **dd,
            )
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                if getattr(new_module, 'head_hidden_size', 0) == new_module.num_features:
                    new_module.head_hidden_size = num_features
                new_module.num_features = num_features

    new_module.eval()
    parent_module.eval()

    # Rebuilding the layers above changes their output channel counts, but any
    # feature-extraction metadata (`feature_info`) still carries the original,
    # unpruned channel counts. Recompute it from the pruned modules so that
    # `feature_info.channels()` and `features_only=True` report correct values.
    _adapt_feature_info(new_module)

    return new_module


def _adapt_feature_info(module):
    """Recompute a pruned model's ``feature_info`` channel counts in-place.

    ``adapt_model_from_string`` rebuilds Conv/BN/Linear layers with pruned
    dimensions but does not touch ``feature_info``, which is populated at
    original build time with the unpruned channel counts. This runs a single
    dry-run forward with hooks on the feature modules to read their true output
    channel counts and writes them back. Any failure leaves ``feature_info``
    untouched; pruning itself is unaffected.
    """
    feature_info = getattr(module, 'feature_info', None)
    if not feature_info:
        return

    # `feature_info` is either a plain list of dicts (during model init) or a
    # `FeatureInfo` wrapper; in both cases we want the underlying list of dicts.
    info_dicts = feature_info.info if hasattr(feature_info, 'info') else feature_info
    if not isinstance(info_dicts, (list, tuple)):
        return

    captured = {}
    handles = []

    def _make_hook(idx):
        def _hook(_mod, _inp, out):
            if isinstance(out, torch.Tensor) and out.ndim >= 2:
                captured[idx] = out.shape[1]
        return _hook

    was_training = module.training
    try:
        for idx, info in enumerate(info_dicts):
            layer_name = info.get('module', '') if isinstance(info, dict) else ''
            if not layer_name:
                continue
            target = extract_layer(module, layer_name)
            if isinstance(target, nn.Module):
                handles.append(target.register_forward_hook(_make_hook(idx)))
        if not handles:
            return

        param = next(module.parameters(), None)
        if param is None:
            return

        # Input channels must match the (possibly pruned / in_chans-adjusted)
        # stem; spatial size comes from the model config when available.
        in_chans = 3
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                in_chans = m.in_channels
                break
        input_size = (in_chans, 224, 224)
        cfg = getattr(module, 'pretrained_cfg', None) or getattr(module, 'default_cfg', None)
        if isinstance(cfg, dict) and cfg.get('input_size'):
            input_size = (in_chans,) + tuple(cfg['input_size'][1:])

        module.eval()
        with torch.no_grad():
            module(torch.zeros(1, *input_size, device=param.device, dtype=param.dtype))
    except Exception as e:  # noqa: BLE001 - metadata recompute must never break model construction
        _logger.warning('Could not recompute pruned feature_info channels: %s', e)
        return
    finally:
        for h in handles:
            h.remove()
        module.train(was_training)

    for idx, num_chs in captured.items():
        if isinstance(info_dicts[idx], dict):
            info_dicts[idx]['num_chs'] = num_chs


def adapt_model_from_file(parent_module, model_variant):
    """Adapt a model to pruned structure from file specification.

    Args:
        parent_module: Original model to adapt.
        model_variant: Name of pruned model variant file.

    Returns:
        Adapted model with pruned layer dimensions.
    """
    adapt_data = pkgutil.get_data(__name__, os.path.join('_pruned', model_variant + '.txt'))
    return adapt_model_from_string(parent_module, adapt_data.decode('utf-8').strip())
