"""Training loop utilities for timm engine.

Provides the core training loop functionality including gradient accumulation,
AMP support, and distributed training handling.
"""
import logging
import time
from collections import OrderedDict
from contextlib import suppress
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from timm.models import model_parameters
from timm.utils import AverageMeter, dispatch_clip_grad, reduce_tensor

from .config import TrainConfig
from .device import DeviceEnv, is_primary

_logger = logging.getLogger(__name__)

# Optional: try to import torchvision for save_images
try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def train_one_epoch(
        epoch: int,
        task: nn.Module,
        loader: DataLoader,
        optimizer: Optimizer,
        device_env: DeviceEnv,
        cfg: TrainConfig,
        lr_scheduler: Optional[Any] = None,
        model_ema: Optional[Any] = None,
        mixup_fn: Optional[Callable] = None,
        saver: Optional[Any] = None,
        output_dir: Optional[str] = None,
        num_updates_total: Optional[int] = None,
        naflex_mode: bool = False,
) -> OrderedDict:
    """Run one epoch of training.

    This function handles:
    - Gradient accumulation
    - AMP forward/backward passes
    - Loss scaling for float16
    - Gradient clipping
    - EMA model updates
    - Progress logging
    - Recovery checkpoints

    Args:
        epoch: Current epoch number.
        task: Training task (handles forward + loss).
        loader: Training data loader.
        optimizer: Optimizer instance.
        device_env: Device environment.
        cfg: Training configuration.
        lr_scheduler: Learning rate scheduler (optional).
        model_ema: Model EMA instance (optional).
        mixup_fn: Mixup/CutMix function (optional).
        saver: Checkpoint saver (optional).
        output_dir: Output directory for saving images/checkpoints.
        num_updates_total: Total number of updates across all epochs.
        naflex_mode: Whether using NaFlex variable-resolution loader.

    Returns:
        OrderedDict with training metrics (at minimum 'loss').

    Example::

        train_metrics = train_one_epoch(
            epoch=0,
            task=task,
            loader=train_loader,
            optimizer=optimizer,
            device_env=device_env,
            cfg=cfg,
            lr_scheduler=scheduler,
            model_ema=model_ema,
        )
    """
    # Handle mixup off epoch
    if cfg.mixup.mixup_off_epoch and epoch >= cfg.mixup.mixup_off_epoch:
        if cfg.loader.prefetcher and hasattr(loader, 'mixup_enabled'):
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(task, 'no_sync')
    update_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    # Get the underlying model for no_sync
    model = task

    model.train()

    accum_steps = cfg.model.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0

    amp_autocast = device_env.amp_autocast
    loss_scaler = device_env.loss_scaler
    device = device_env.device
    model_dtype = device_env.model_dtype

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not cfg.loader.prefetcher:
            input = input.to(device=device, dtype=model_dtype)
            target = target.to(device=device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        if cfg.device.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # Multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                result = task(input, target)
                _loss = result['loss']

            if accum_steps > 1:
                _loss = _loss / accum_steps
            return _loss, result

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=cfg.optimizer.clip_grad,
                    clip_mode=cfg.optimizer.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in cfg.optimizer.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if cfg.optimizer.clip_grad is not None:
                        dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in cfg.optimizer.clip_mode),
                            value=cfg.optimizer.clip_grad,
                            mode=cfg.optimizer.clip_mode,
                        )
                    optimizer.step()

        if naflex_mode:
            assert isinstance(input, dict)
            batch_size = input['patches'].shape[0]

            # Scale gradient vs the minimum batch size
            naflex_loss_scale = cfg.naflex.naflex_loss_scale
            if not naflex_loss_scale or naflex_loss_scale == 'none':
                local_scale = 1.0
            else:
                local_scale = batch_size / cfg.loader.batch_size
                if naflex_loss_scale == 'sqrt':
                    local_scale = local_scale ** 0.5

            if device_env.distributed:
                global_batch_size = reduce_tensor(
                    torch.tensor(batch_size, device=device, dtype=torch.float32),
                    1  # SUM
                )
                dist_scale = device_env.world_size * batch_size / global_batch_size
            else:
                dist_scale = None
                global_batch_size = batch_size

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, result = _forward()
                    scaled_loss = local_scale * loss
                    if dist_scale is not None:
                        scaled_loss = scaled_loss * dist_scale
                    _backward(scaled_loss)
            else:
                loss, result = _forward()
                scaled_loss = local_scale * loss
                if dist_scale is not None:
                    scaled_loss = scaled_loss * dist_scale
                _backward(scaled_loss)
        else:
            global_batch_size = batch_size = input.shape[0]
            if device_env.distributed:
                global_batch_size *= device_env.world_size

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, result = _forward()
                    _backward(loss)
            else:
                loss, result = _forward()
                _backward(loss)

        losses_m.update(loss.item() * accum_steps, batch_size)
        update_sample_count += global_batch_size

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()

        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if cfg.device.synchronize_step:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'npu':
                torch.npu.synchronize()

        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % cfg.misc.log_interval == 0 or last_batch:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val
            if device_env.distributed:
                loss_avg = reduce_tensor(
                    loss.new([loss_avg]), device_env.world_size
                ).item()
                loss_now = reduce_tensor(
                    loss.new([loss_now]), device_env.world_size
                ).item()

            if is_primary(device_env):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                    f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

                if cfg.misc.save_images and output_dir and HAS_TORCHVISION:
                    import os
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, f'train-batch-{batch_idx}.jpg'),
                        padding=0,
                        normalize=True,
                    )

        if saver is not None and cfg.misc.recovery_interval and (
                (update_idx + 1) % cfg.misc.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()

    # End of epoch
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    loss_avg = losses_m.avg
    if device_env.distributed:
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = reduce_tensor(loss_avg, device_env.world_size).item()

    return OrderedDict([('loss', loss_avg)])
