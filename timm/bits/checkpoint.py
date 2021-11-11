import logging
import os
from collections import OrderedDict
from typing import Dict, Any, Callable

import torch

from timm.utils import unwrap_model

from .device_env import DeviceEnv
from .train_state import TrainState

_logger = logging.getLogger(__name__)


def save_train_state(
        checkpoint_path: str,  # FIXME pass base path + file pattern + epoch / step separately for DS?
        train_state: TrainState,
        extra_state: Dict[str, Any] = None,
        unwrap_fn: Callable = unwrap_model,
        dev_env: DeviceEnv = None,
        log_info: bool = True):

    assert not train_state.updater.deepspeed
    # DeepSpeed has a fully custom checkpoint saving setup, it is not possible
    # specify a filename, checkpoints needed to be saved from all ranks, etc
    # if train_state.updater.deepspeed:
    #     save_train_state_deepspeed(train_state, checkpoint_path)

    dev_env = dev_env or DeviceEnv.instance()
    state_dict = train_state.state_dict(unwrap_fn=unwrap_fn)
    if extra_state:
        state_dict.update(extra_state)
    if dev_env.type_xla:
        # XLA state dict needs to be moved to CPU before save, this is normally done by xm.save
        state_dict = dev_env.state_dict_to_cpu(state_dict)
    torch.save(state_dict, checkpoint_path)


def load_train_state(
        train_state: TrainState,
        checkpoint_path: str,  # FIXME pass base path + file pattern + epoch / step separately for DS
        unwrap_fn: Callable = None,
        load_opt: bool = True,
        dev_env: DeviceEnv = None,
        log_info: bool = True
):
    unwrap_fn = unwrap_fn or unwrap_model
    if not os.path.isfile(checkpoint_path):
        _logger.error("No valid resume checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

    if log_info:
        _logger.info('Restoring training state from checkpoint...')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    assert isinstance(checkpoint, dict)

    if not checkpoint.get('version', 0) > 2:
        load_legacy_checkpoint(train_state, checkpoint=checkpoint, load_opt=load_opt, log_info=log_info)
        if log_info:
            _logger.info("Loaded legacy checkpoint '{}' (epoch {})".format(checkpoint_path, train_state.epoch))
        return

    train_state.load_state_dict(checkpoint, unwrap_fn=unwrap_fn, load_opt=load_opt)
    if log_info:
        _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, train_state.epoch))


def _get_state_dict(checkpoint, state_dict_key='state_dict'):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[state_dict_key].items():
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v
    return new_state_dict


def load_legacy_checkpoint(
        train_state: TrainState,
        checkpoint,
        load_opt=True,
        log_info=True):

    assert isinstance(checkpoint, dict) and 'state_dict' in checkpoint
    train_state.model.load_state_dict(_get_state_dict(checkpoint))

    if train_state.model_ema is not None and 'state_dict_ema' in checkpoint:
        if log_info:
            _logger.info('Restoring model (EMA) state from checkpoint...')
        unwrap_model(train_state.model_ema).load_state_dict(_get_state_dict(checkpoint, 'state_dict_ema'))

    if load_opt:
        if train_state.updater.optimizer is not None and 'optimizer' in checkpoint:
            if log_info:
                _logger.info('Restoring optimizer state from checkpoint...')
            train_state.updater.optimizer.load_state_dict(checkpoint['optimizer'])

        scaler_state_dict_key = 'amp_scaler'
        if train_state.updater.grad_scaler is not None and scaler_state_dict_key in checkpoint:
            if log_info:
                _logger.info('Restoring AMP loss scaler state from checkpoint...')
            train_state.updater.grad_scaler.load_state_dict(checkpoint[scaler_state_dict_key])

    if 'epoch' in checkpoint:
        resume_epoch = checkpoint['epoch']
        if 'version' in checkpoint and checkpoint['version'] > 1:
            resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
        train_state.epoch = resume_epoch  # FIXME use replace if we make train_state read-only

