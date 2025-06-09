from dataclasses import replace as _rep
from functools import partial
import math

import torch.nn as nn
import timm.models.cspnet as _cspnet
from timm.layers import get_act_layer
from timm.models.ghostnet import GhostBottleneck

from .cspnet import (
    CspNet, _create_cspnet, _cfg,
    register_model, generate_default_cfgs, model_cfgs as _base_cfgs
)

def _ghost_adapter(
    *,
    in_chs: int,
    out_chs: int,
    bottle_ratio: float = 0.25,
    dw_kernel_size: int = 3,
    stride: int = 1,
    act_layer=nn.ReLU,
    se_ratio: float = 0.,
    use_ghost: bool = True,
    mode: str = "original",
    **_ignored,
):
    if isinstance(act_layer, str):
        act_layer = get_act_layer(act_layer)

    mid_chs = int(math.ceil(out_chs * bottle_ratio))

    if not use_ghost:
        mode = "plain"

    return GhostBottleneck(
        in_chs=in_chs,
        mid_chs=mid_chs,
        out_chs=out_chs,
        dw_kernel_size=dw_kernel_size,
        stride=stride,
        act_layer=act_layer,
        se_ratio=se_ratio,
        mode=mode,
    )

def _make_get_block_fn(use_ghost_flag: bool):
    from timm.models.cspnet import _get_block_fn as _orig

    def _patched(stage_args):
        block_type = stage_args.pop("block_type")
        if block_type == "ghost":
            return partial(_ghost_adapter, use_ghost=use_ghost_flag), stage_args
        return _orig(stage_args)

    return _patched

model_cfgs = dict(
    cspghostnet=_rep(
        _base_cfgs["cspresnet50"],
        stages=_rep(
            _base_cfgs["cspresnet50"].stages,
            block_type="ghost",
        ),
    )
)
_cspnet.model_cfgs.update(model_cfgs)

@register_model
def cspghostnet(pretrained: bool = False, **kwargs) -> CspNet:
    use_ghost_flag = kwargs.pop("use_ghost", True)

    _cspnet._get_block_fn = _make_get_block_fn(use_ghost_flag)

    model = _create_cspnet("cspghostnet", pretrained=pretrained, **kwargs)

    return model

default_cfgs = generate_default_cfgs({
    "cspghostnet.untrained": _cfg(),
})