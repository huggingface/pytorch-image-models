"""Checkpoint helpers for task-based training."""
import argparse
import logging
import os
from typing import Optional

import torch

from timm.models import clean_state_dict


_logger = logging.getLogger(__name__)


def _load_train_checkpoint(checkpoint_path, weights_only=True):
    use_safe_globals = weights_only and hasattr(torch.serialization, 'safe_globals')
    if use_safe_globals:
        with torch.serialization.safe_globals([argparse.Namespace]):
            return torch.load(checkpoint_path, map_location='cpu', weights_only=weights_only)
    return torch.load(checkpoint_path, map_location='cpu', weights_only=weights_only)


def resume_task_checkpoint(
        task,
        checkpoint_path,
        optimizer=None,
        loss_scaler=None,
        log_info=True,
        weights_only=True,
) -> Optional[int]:
    """Resume a task-based training checkpoint.

    Supports task checkpoints with ``state_dict``/``task_state`` and legacy
    training checkpoints that used a bare ``model`` key.
    """
    resume_epoch = None
    if not os.path.isfile(checkpoint_path):
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

    checkpoint = _load_train_checkpoint(checkpoint_path, weights_only=weights_only)
    if isinstance(checkpoint, dict):
        state_dict_key = ''
        task_state_key = ''
        if 'state_dict' in checkpoint:
            state_dict_key = 'state_dict'
            task_state_key = 'task_state'
        elif 'model' in checkpoint:
            state_dict_key = 'model'

        if state_dict_key:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            task.load_checkpoint_state(
                clean_state_dict(checkpoint[state_dict_key]),
                checkpoint.get(task_state_key) if task_state_key else None,
            )

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
                    resume_epoch += 1
                if log_info:
                    _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            return resume_epoch

    task.load_checkpoint_state(clean_state_dict(checkpoint))
    if log_info:
        _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
    return resume_epoch


def load_task_ema_checkpoint(task, checkpoint_path, weights_only=True):
    """Load EMA weights and optional task state into a task EMA module."""
    checkpoint = _load_train_checkpoint(checkpoint_path, weights_only=weights_only)
    state_dict_key = ''
    task_state_key = ''
    if isinstance(checkpoint, dict):
        if checkpoint.get('state_dict_ema', None) is not None:
            state_dict_key = 'state_dict_ema'
            task_state_key = 'task_state_ema'
        elif checkpoint.get('model_ema', None) is not None:
            state_dict_key = 'model_ema'
        elif 'state_dict' in checkpoint:
            state_dict_key = 'state_dict'
            task_state_key = 'task_state'
        elif 'model' in checkpoint:
            state_dict_key = 'model'

    state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
    task.load_checkpoint_state(
        state_dict,
        checkpoint.get(task_state_key) if isinstance(checkpoint, dict) and task_state_key else None,
        ema=True,
    )
    _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key or 'checkpoint', checkpoint_path))
