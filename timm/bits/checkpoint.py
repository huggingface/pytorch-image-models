import logging
import os
from collections import OrderedDict

import torch

from timm.utils import unwrap_model

from .train_state import TrainState, serialize_train_state, deserialize_train_state

_logger = logging.getLogger(__name__)


def _load_state_dict(checkpoint, state_dict_key='state_dict'):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[state_dict_key].items():
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v
    return new_state_dict


def resume_train_checkpoint(
        train_state: TrainState,
        checkpoint_path,
        resume_opt=True,
        deserialize_fn=deserialize_train_state,
        log_info=True):

    # FIXME this is a hacky adaptation of pre-bits resume to get up and running quickly
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict) and 'state_dict' in checkpoint
        if log_info:
            _logger.info('Restoring model state from checkpoint...')

        train_state.model.load_state_dict(_load_state_dict(checkpoint))

        if train_state.model_ema is not None and 'state_dict_ema' in checkpoint:
            if log_info:
                _logger.info('Restoring model (EMA) state from checkpoint...')
            unwrap_model(train_state.model_ema).load_state_dict(_load_state_dict(checkpoint, 'state_dict_ema'))

        if resume_opt:
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

        if log_info:
            _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        _logger.error("No valid resume checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
