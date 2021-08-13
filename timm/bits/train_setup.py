import dataclasses
from typing import Callable, Union, Optional
import logging

import torch
import torch.nn as nn

from timm.optim import create_optimizer_v2
from timm.utils import ModelEmaV2

try:
    import deepspeed as ds
except ImportError:
    ds = None

from .checkpoint import load_train_state
from .device_env import DeviceEnv
from .train_cfg import TrainCfg
from .train_state import TrainState
from .updater_factory import create_updater


_logger = logging.getLogger(__name__)


def setup_model_and_optimizer(
        dev_env: DeviceEnv,
        model: nn.Module,
        optimizer: Union[Callable, str],
        optimizer_cfg,
        clip_fn: Optional[Union[Callable, str]] = None,
        clip_value: Optional[float] = None,
        model_ema: bool = False,
        model_ema_decay: float = 0.9999,
        use_syncbn: bool = False,
        resume_path: str = '',
        resume_opt: bool = True,
        deepspeed: bool = False,
):
    """

    Args:
        dev_env:
        model:
        optimizer:
        optimizer_cfg:
        clip_value:
        clip_fn:
        model_ema:
        model_ema_decay:
        use_syncbn:
        resume_path:
        resume_opt:

    Returns:

    """
    if deepspeed:
        return setup_model_and_optimizer_deepspeed(
            dev_env=dev_env, model=model, optimizer=optimizer, optimizer_cfg=optimizer_cfg,
            clip_fn=clip_fn, clip_value=clip_value, model_ema=model_ema, model_ema_decay=model_ema_decay,
            resume_path=resume_path, resume_opt=resume_opt,
        )

    dev_env.to_device(model)

    if use_syncbn and dev_env.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if dev_env.primary:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if isinstance(optimizer, Callable):
        optimizer = optimizer(model=model, **optimizer_cfg)
    else:
        optimizer = create_optimizer_v2(model=model, **optimizer_cfg)

    updater = create_updater(
        model=model,
        optimizer=optimizer,
        clip_fn=clip_fn,
        clip_value=clip_value,
    )

    # ema model
    model_ema = ModelEmaV2(model, decay=model_ema_decay) if model_ema else None

    train_state = TrainState(model=model, updater=updater, model_ema=model_ema)

    if resume_path:
        load_train_state(
            train_state,
            resume_path,
            load_opt=resume_opt,
            log_info=dev_env.primary)

    if dev_env.distributed:
         train_state = dataclasses.replace(
             train_state, model=dev_env.wrap_distributed(train_state.model))

    return train_state


def setup_model_and_optimizer_deepspeed(
        dev_env: DeviceEnv,
        model: nn.Module,
        optimizer: Union[Callable, str],
        optimizer_cfg,
        clip_fn: Optional[Union[Callable, str]] = None,
        clip_value: Optional[float] = None,
        model_ema: bool = False,
        model_ema_decay: float = 0.9999,
        use_syncbn: bool = False,
        resume_path: str = '',
        resume_opt: bool = True,
):
    dev_env.to_device(model)

    if isinstance(optimizer, Callable):
        optimizer = optimizer(model=model, **optimizer_cfg)
    else:
        optimizer = create_optimizer_v2(model=model, **optimizer_cfg)

    model = ds.initialize(model=model, optimizer=optimizer, dist_init_required=False)

    updater = create_updater(
        model=model,
        optimizer=optimizer,
        clip_fn=clip_fn,
        clip_value=clip_value,
        deepspeed=True,
    )

    # ema model
    # FIXME how to do EMA w/ deepspeed?
    model_ema = ModelEmaV2(model, decay=model_ema_decay) if model_ema else None

    train_state = TrainState(model=model, updater=updater, model_ema=model_ema)

    if resume_path:
        # FIXME deepspeed resumes differently
        assert False

    if dev_env.distributed:
         train_state = dataclasses.replace(
             train_state, model=dev_env.wrap_distributed(train_state.model))

    return train_state
