import fnmatch
import logging
from itertools import islice
from typing import Collection, Optional

from torch import nn as nn

from timm.models import group_parameters


_logger = logging.getLogger(__name__)


def _matches_pattern(name: str, patterns: Collection[str]) -> bool:
    """Check if parameter name matches any pattern (supports wildcards)."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def param_groups_weight_decay(
        model: nn.Module,
        weight_decay: float = 1e-5,
        no_weight_decay_list: Collection[str] = (),
        fallback_list: Collection[str] = (),
        fallback_no_weight_decay: bool = False,
):
    # Merge no_weight_decay into fallback_list if requested
    if fallback_no_weight_decay:
        fallback_list = set(fallback_list) | set(no_weight_decay_list)

    decay = []
    decay_fallback = []
    no_decay = []
    no_decay_fallback = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if this is a "fallback" parameter for fallback optimizer (if available)
        is_fallback = _matches_pattern(name, fallback_list)

        # Determine weight decay
        matches_pattern = _matches_pattern(name, no_weight_decay_list)
        if param.ndim <= 1 or name.endswith(".bias") or matches_pattern:
            # No weight decay
            if is_fallback:
                no_decay_fallback.append(param)
            else:
                no_decay.append(param)
        else:
            # With weight decay
            if is_fallback:
                decay_fallback.append(param)
            else:
                decay.append(param)

    groups = []
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.})
    if decay:
        groups.append({'params': decay, 'weight_decay': weight_decay})
    if no_decay_fallback:
        groups.append({'params': no_decay_fallback, 'weight_decay': 0., 'use_fallback': True})
    if decay_fallback:
        groups.append({'params': decay_fallback, 'weight_decay': weight_decay, 'use_fallback': True})

    return groups

def _group(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def auto_group_layers(model, layers_per_group=12, num_groups=None):
    def _in_head(n, hp):
        if not hp:
            return True
        elif isinstance(hp, (tuple, list)):
            return any([n.startswith(hpi) for hpi in hp])
        else:
            return n.startswith(hp)

    head_prefix = getattr(model, 'pretrained_cfg', {}).get('classifier', None)
    names_trunk = []
    names_head = []
    for n, _ in model.named_parameters():
        names_head.append(n) if _in_head(n, head_prefix) else names_trunk.append(n)

    # group non-head layers
    num_trunk_layers = len(names_trunk)
    if num_groups is not None:
        layers_per_group = -(num_trunk_layers // -num_groups)
    names_trunk = list(_group(names_trunk, layers_per_group))

    num_trunk_groups = len(names_trunk)
    layer_map = {n: i for i, l in enumerate(names_trunk) for n in l}
    layer_map.update({n: num_trunk_groups for n in names_head})
    return layer_map

_layer_map = auto_group_layers  # backward compat


def param_groups_layer_decay(
        model: nn.Module,
        weight_decay: float = 0.05,
        no_weight_decay_list: Collection[str] = (),
        fallback_list: Collection[str] = (),
        fallback_no_weight_decay: bool = False,
        weight_decay_exclude_1d: bool = True,
        layer_decay: float = .75,
        min_scale: float = 0.,
        no_opt_scale: Optional[float] = None,
        verbose: bool = False,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    # Merge no_weight_decay into fallback_list if requested
    if fallback_no_weight_decay:
        fallback_list = set(fallback_list) | set(no_weight_decay_list)

    param_group_names = {}  # NOTE for debugging
    param_groups = {}

    if hasattr(model, 'group_matcher'):
        # FIXME interface needs more work
        layer_map = group_parameters(model, model.group_matcher(coarse=False), reverse=True)
    else:
        # fallback
        layer_map = auto_group_layers(model)
    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(max(min_scale, layer_decay ** (layer_max - i)) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if this is a "fallback" parameter for fallback optimizer (if available)
        is_fallback = _matches_pattern(name, fallback_list)

        # Determine weight decay
        if (weight_decay_exclude_1d and param.ndim <= 1) or _matches_pattern(name, no_weight_decay_list):
            # no weight decay for 1D parameters and model specific ones
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        this_scale = layer_scales[layer_id]
        if no_opt_scale and this_scale < no_opt_scale:
            # if the calculated layer scale is below this, exclude from optimization
            param.requires_grad = False
            continue

        fallback_suffix = "_fallback" if is_fallback else ""
        group_name = "layer_%d_%s%s" % (layer_id, g_decay, fallback_suffix)

        if group_name not in param_groups:
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "use_fallback": is_fallback,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            if is_fallback:
                param_groups[group_name]["use_fallback"] = True

        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    if verbose:
        import json
        _logger.info("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())
