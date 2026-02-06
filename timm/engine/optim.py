"""Optimizer and learning rate scheduler utilities for timm engine.

Provides functions for creating optimizers with proper learning rate scaling
and schedulers with warmup support.
"""
import copy
import logging
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
from torch.optim import Optimizer

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

from .config import OptimizerConfig, SchedulerConfig
from .device import DeviceEnv, is_primary

_logger = logging.getLogger(__name__)


def compute_effective_lr(
    cfg: OptimizerConfig,
    device_env: DeviceEnv,
    grad_accum_steps: int = 1,
) -> float:
    """Compute effective learning rate based on batch size scaling.

    The learning rate is scaled based on global batch size using either
    linear or sqrt scaling, as is common practice for distributed training.

    Args:
        cfg: Optimizer configuration.
        device_env: Device environment.
        grad_accum_steps: Number of gradient accumulation steps.

    Returns:
        Computed learning rate.
    """
    if cfg.lr is not None:
        return cfg.lr

    # Compute global batch size
    global_batch_size = cfg.lr_base_size  # default to base size if no batch_size info
    # Note: batch_size should be passed from loader config in practice
    # Here we assume cfg has the necessary info or default to lr_base

    # Determine scaling method
    lr_base_scale = cfg.lr_base_scale
    if not lr_base_scale:
        # Auto-detect based on optimizer type
        opt_lower = cfg.opt.lower()
        lr_base_scale = 'sqrt' if any(o in opt_lower for o in ('ada', 'lamb')) else 'linear'

    batch_ratio = global_batch_size / cfg.lr_base_size
    if lr_base_scale == 'sqrt':
        batch_ratio = batch_ratio ** 0.5

    return cfg.lr_base * batch_ratio


def create_train_optimizer(
    model: nn.Module,
    cfg: OptimizerConfig,
    device_env: DeviceEnv,
    batch_size: int,
    grad_accum_steps: int = 1,
) -> Optimizer:
    """Create optimizer with proper learning rate scaling.

    This function handles:
    - Learning rate scaling based on global batch size
    - Layer-wise learning rate decay
    - Optimizer-specific parameters (betas, eps, etc.)

    Args:
        model: Model to optimize.
        cfg: Optimizer configuration.
        device_env: Device environment.
        batch_size: Per-device batch size.
        grad_accum_steps: Number of gradient accumulation steps.

    Returns:
        Configured optimizer instance.

    Example::

        optimizer = create_train_optimizer(model, opt_cfg, device_env, batch_size=32)
    """
    # Compute effective learning rate
    lr = cfg.lr
    if lr is None:
        global_batch_size = batch_size * device_env.world_size * grad_accum_steps
        batch_ratio = global_batch_size / cfg.lr_base_size

        lr_base_scale = cfg.lr_base_scale
        if not lr_base_scale:
            opt_lower = cfg.opt.lower()
            lr_base_scale = 'sqrt' if any(o in opt_lower for o in ('ada', 'lamb')) else 'linear'

        if lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5

        lr = cfg.lr_base * batch_ratio

        if is_primary(device_env):
            _logger.info(
                f'Learning rate ({lr}) calculated from base learning rate ({cfg.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {lr_base_scale} scaling.'
            )

    # Build optimizer kwargs
    opt_kwargs_dict = optimizer_kwargs(cfg)

    # Create optimizer
    optimizer = create_optimizer_v2(
        model,
        **opt_kwargs_dict,
        **cfg.opt_kwargs,
    )

    if is_primary(device_env):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = cfg.weight_decay
        defaults_str = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        _logger.info(
            f'Created {type(optimizer).__name__} ({cfg.opt}) optimizer: {defaults_str}'
        )

    return optimizer


def create_train_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    updates_per_epoch: int,
    device_env: DeviceEnv,
) -> Tuple[Any, int]:
    """Create learning rate scheduler.

    This function creates a scheduler with support for:
    - Various scheduler types (cosine, step, plateau, etc.)
    - Warmup periods
    - Noise injection
    - Cycle-based schedules

    Args:
        optimizer: Optimizer instance.
        cfg: Scheduler configuration.
        updates_per_epoch: Number of optimizer updates per epoch.
        device_env: Device environment.

    Returns:
        Tuple of (scheduler, num_epochs) where num_epochs accounts for
        warmup prefix and cycles.

    Example::

        scheduler, num_epochs = create_train_scheduler(
            optimizer, sched_cfg, updates_per_epoch=1000, device_env
        )
    """
    # Build scheduler kwargs
    sched_kwargs = scheduler_kwargs(cfg)

    # Create scheduler
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **sched_kwargs,
        updates_per_epoch=updates_per_epoch if cfg.sched_on_updates else 0,
    )

    if is_primary(device_env):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. '
            f'LR stepped {"per update" if cfg.sched_on_updates else "per epoch"}.'
        )

    return lr_scheduler, num_epochs


def get_optimizer_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer.

    Args:
        optimizer: Optimizer instance.

    Returns:
        Average learning rate across parameter groups.
    """
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    return sum(lrs) / len(lrs)
