"""Checkpoint management utilities for timm engine.

Provides functions for saving and loading checkpoints, and setting up
checkpoint savers for training.
"""
import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from timm.models import load_checkpoint, resume_checkpoint
from timm.utils import CheckpointSaver

from .config import TrainConfig
from .device import DeviceEnv, is_primary

_logger = logging.getLogger(__name__)


def setup_checkpoint_saver(
    model: nn.Module,
    optimizer: Optimizer,
    cfg: TrainConfig,
    output_dir: str,
    device_env: DeviceEnv,
    model_ema: Optional[Any] = None,
    decreasing_metric: bool = False,
) -> Optional[CheckpointSaver]:
    """Setup checkpoint saver for training.

    Creates a CheckpointSaver instance configured for saving model checkpoints
    during training. Only creates a saver on the primary process.

    Supports both patterns:
    - Old: pass model + model_ema separately
    - New: pass task (extracts trainable_module and trainable_module_ema)

    Args:
        model: Model or training task to save. If a TrainingTask with
            trainable_module attribute, extracts that for checkpointing.
        optimizer: Optimizer state to save.
        cfg: Training configuration.
        output_dir: Directory to save checkpoints.
        device_env: Device environment.
        model_ema: Model EMA instance (optional). If None and model is a
            TrainingTask with trainable_module_ema, uses that instead.
        decreasing_metric: If True, lower metric values are better.

    Returns:
        CheckpointSaver instance on primary process, None otherwise.

    Example::

        # Old pattern (external EMA)
        saver = setup_checkpoint_saver(
            model, optimizer, cfg, output_dir, device_env,
            model_ema=model_ema, decreasing_metric=False
        )

        # New pattern (task-based EMA)
        saver = setup_checkpoint_saver(
            task, optimizer, cfg, output_dir, device_env,
            decreasing_metric=False
        )
    """
    if not is_primary(device_env):
        return None

    # Extract model and EMA from task if it follows TrainingTask pattern
    if model_ema is None and hasattr(model, 'trainable_module'):
        # Task-based pattern: extract components from task
        actual_model = model.trainable_module
        actual_ema = getattr(model, 'trainable_module_ema', None)
    else:
        # Old pattern: use model and model_ema as provided
        actual_model = model
        actual_ema = model_ema

    saver_kwargs = dict(
        model=actual_model,
        optimizer=optimizer,
        args=None,  # We pass config differently
        model_ema=actual_ema,
        amp_scaler=device_env.loss_scaler,
        checkpoint_dir=output_dir,
        recovery_dir=output_dir,
        decreasing=decreasing_metric,
        max_history=cfg.misc.checkpoint_hist,
    )

    saver = CheckpointSaver(**saver_kwargs)

    return saver


def resume_training(
    model: nn.Module,
    optimizer: Optimizer,
    resume_path: str,
    device_env: DeviceEnv,
    no_resume_opt: bool = False,
) -> Optional[int]:
    """Resume training from a checkpoint.

    Loads model and optionally optimizer state from a checkpoint file.

    Args:
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        resume_path: Path to checkpoint file.
        device_env: Device environment.
        no_resume_opt: If True, don't resume optimizer state.

    Returns:
        Resume epoch number, or None if not resuming.

    Example::

        resume_epoch = resume_training(
            model, optimizer, 'checkpoint.pth', device_env
        )
        start_epoch = resume_epoch if resume_epoch is not None else 0
    """
    if not resume_path:
        return None

    resume_epoch = resume_checkpoint(
        model,
        resume_path,
        optimizer=None if no_resume_opt else optimizer,
        loss_scaler=None if no_resume_opt else device_env.loss_scaler,
        log_info=is_primary(device_env),
    )

    return resume_epoch


def load_pretrained(
    model: nn.Module,
    checkpoint_path: str,
    device_env: DeviceEnv,
    strict: bool = False,
) -> None:
    """Load pretrained weights into model.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        device_env: Device environment.
        strict: If True, require exact key match.
    """
    load_checkpoint(
        model,
        checkpoint_path,
        strict=strict,
    )

    if is_primary(device_env):
        _logger.info(f'Loaded pretrained weights from: {checkpoint_path}')


def save_config(cfg: TrainConfig, output_dir: str, filename: str = 'config.yaml') -> str:
    """Save training configuration to file.

    Args:
        cfg: Training configuration.
        output_dir: Directory to save config.
        filename: Config filename.

    Returns:
        Path to saved config file.
    """
    import dataclasses
    import yaml

    config_path = os.path.join(output_dir, filename)

    # Convert dataclass to dict recursively
    def dataclass_to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [dataclass_to_dict(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: dataclass_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    config_dict = dataclass_to_dict(cfg)

    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return config_path


def get_output_dir(base_output: str, experiment_name: str) -> str:
    """Get output directory for experiment.

    Creates the directory if it doesn't exist.

    Args:
        base_output: Base output directory.
        experiment_name: Experiment name (subdirectory).

    Returns:
        Full path to output directory.
    """
    from timm.utils import get_outdir
    return get_outdir(base_output or './output/train', experiment_name)
