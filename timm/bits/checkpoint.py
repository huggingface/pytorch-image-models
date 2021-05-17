import logging
import os
from collections import OrderedDict

import torch

from .train_state import TrainState, serialize_train_state, deserialize_train_state

_logger = logging.getLogger(__name__)


def resume_train_checkpoint(
        train_state,
        checkpoint_path,
        resume_opt=True,
        deserialize_fn=deserialize_train_state,
        log_info=True):

    raise NotImplementedError

    # resume_epoch = None
    # if os.path.isfile(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #
    #     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    #         if log_info:
    #             _logger.info('Restoring model state from checkpoint...')
    #         new_state_dict = OrderedDict()
    #         for k, v in checkpoint['state_dict'].items():
    #             name = k[7:] if k.startswith('module') else k
    #             new_state_dict[name] = v
    #         model.load_state_dict(new_state_dict)
    #
    #         if optimizer is not None and 'optimizer' in checkpoint:
    #             if log_info:
    #                 _logger.info('Restoring optimizer state from checkpoint...')
    #             optimizer.load_state_dict(checkpoint['optimizer'])
    #
    #         if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
    #             if log_info:
    #                 _logger.info('Restoring AMP loss scaler state from checkpoint...')
    #             loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])
    #
    #         if 'epoch' in checkpoint:
    #             resume_epoch = checkpoint['epoch']
    #             if 'version' in checkpoint and checkpoint['version'] > 1:
    #                 resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
    #
    #         if log_info:
    #             _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    #     else:
    #         model.load_state_dict(checkpoint)
    #         if log_info:
    #             _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
    #     return resume_epoch
    # else:
    #     _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
    #     raise FileNotFoundError()
