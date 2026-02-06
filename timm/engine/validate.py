"""Validation utilities for timm engine.

Provides the validation loop for evaluating model performance.
"""
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from timm.utils import AverageMeter, accuracy, reduce_tensor

from .config import TrainConfig
from .device import DeviceEnv, is_primary

if TYPE_CHECKING:
    from timm.task import EvalTask

_logger = logging.getLogger(__name__)


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device_env: DeviceEnv,
    cfg: TrainConfig,
    task: Optional[nn.Module] = None,
    log_suffix: str = '',
) -> OrderedDict:
    """Run validation on the evaluation dataset.

    This function handles:
    - Standard classification evaluation
    - Self-supervised evaluation (loss only)
    - Test-time augmentation (TTA)
    - Distributed metric aggregation

    Args:
        model: Model to evaluate.
        loader: Evaluation data loader.
        loss_fn: Loss function.
        device_env: Device environment.
        cfg: Training configuration.
        task: Training task (used for SSL evaluation).
        log_suffix: Suffix for log messages (e.g., ' (EMA)').

    Returns:
        OrderedDict with metrics:
        - For classification: {'loss': float, 'top1': float, 'top5': float}
        - For SSL: {'loss': float}

    Example::

        metrics = validate(
            model=model,
            loader=eval_loader,
            loss_fn=nn.CrossEntropyLoss(),
            device_env=device_env,
            cfg=cfg,
        )
        print(f"Top-1 accuracy: {metrics['top1']:.2f}%")
    """
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    # Check if this is a self-supervised task
    is_ssl = cfg.ssl.ssl_method is not None

    model.eval()

    device = device_env.device
    model_dtype = device_env.model_dtype
    amp_autocast = device_env.amp_autocast

    end = time.time()
    last_idx = len(loader) - 1

    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            if not cfg.loader.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)

            if cfg.device.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                if is_ssl and task is not None:
                    # Self-supervised: use task for forward and loss
                    result = task(input, target)
                    loss = result['loss']
                    output = result.get('output', None)
                else:
                    # Classification: use model directly
                    output = model(input)
                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    # Test-time augmentation reduction
                    reduce_factor = cfg.misc.tta
                    if reduce_factor > 1:
                        output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                        target = target[0:target.size(0):reduce_factor]

                    loss = loss_fn(output, target)

            # Compute accuracy for classification
            if not is_ssl:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # Distributed reduction
            if device_env.distributed:
                reduced_loss = reduce_tensor(loss.data, device_env.world_size)
                if not is_ssl:
                    acc1 = reduce_tensor(acc1, device_env.world_size)
                    acc5 = reduce_tensor(acc5, device_env.world_size)
            else:
                reduced_loss = loss.data

            # Synchronize for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'npu':
                torch.npu.synchronize()

            batch_size = input.shape[0] if not isinstance(input, dict) else input['patches'].shape[0]
            losses_m.update(reduced_loss.item(), batch_size)
            if not is_ssl:
                top1_m.update(acc1.item(), batch_size)
                top5_m.update(acc5.item(), batch_size)

            batch_time_m.update(time.time() - end)
            end = time.time()

            if is_primary(device_env) and (last_batch or batch_idx % cfg.misc.log_interval == 0):
                log_name = 'Test' + log_suffix
                if is_ssl:
                    _logger.info(
                        f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                        f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                        f'Loss: {losses_m.val:>7.4f} ({losses_m.avg:>7.4f})'
                    )
                else:
                    _logger.info(
                        f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                        f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                        f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                        f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                        f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                    )

    if is_ssl:
        metrics = OrderedDict([('loss', losses_m.avg)])
    else:
        metrics = OrderedDict([
            ('loss', losses_m.avg),
            ('top1', top1_m.avg),
            ('top5', top5_m.avg),
        ])

    return metrics


def _device_synchronize(device: torch.device) -> None:
    """Synchronize device for accurate timing."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'npu':
        torch.npu.synchronize()


def validate_with_task(
    eval_task: 'EvalTask',
    loader: DataLoader,
    device_env: DeviceEnv,
    cfg: TrainConfig,
    log_suffix: str = '',
) -> Dict[str, float]:
    """Validation using EvalTask with full engine features.

    Uses the EvalTask pattern (reset → forward → compute_metrics) with
    full engine support for distributed training, timing, and logging.

    Args:
        eval_task: EvalTask instance from task.get_eval_task()
        loader: Evaluation data loader
        device_env: Device environment
        cfg: Training configuration
        log_suffix: Suffix for log messages (e.g., ' (EMA)')

    Returns:
        Final metrics dict from eval_task.compute_metrics()

    Example::

        eval_task = task.get_eval_task(use_ema=True)
        metrics = validate_with_task(eval_task, val_loader, device_env, cfg)
        print(f"Accuracy: {metrics['acc1']:.2f}%")
    """
    batch_time_m = AverageMeter()

    device = device_env.device
    model_dtype = device_env.model_dtype
    amp_autocast = device_env.amp_autocast

    eval_task.reset()

    end = time.time()
    last_idx = len(loader) - 1

    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            if not cfg.loader.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)

            if cfg.device.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                result = eval_task(input, target)

            # Synchronize for accurate timing
            _device_synchronize(device)

            batch_time_m.update(time.time() - end)
            end = time.time()

            # Log progress with running metrics
            if is_primary(device_env) and (last_batch or batch_idx % cfg.misc.log_interval == 0):
                log_name = 'Test' + log_suffix

                # Build log message from available running metrics
                log_parts = [f'{log_name}: [{batch_idx:>4d}/{last_idx}]']
                log_parts.append(f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})')

                # Include running metrics if available
                if 'acc1' in result:
                    log_parts.append(f"Acc@1: {result['acc1'].item():>7.3f}")
                if 'acc5' in result:
                    log_parts.append(f"Acc@5: {result['acc5'].item():>7.3f}")
                if 'num_samples' in result:
                    log_parts.append(f"Samples: {result['num_samples'].item()}")

                _logger.info('  '.join(log_parts))

    metrics = eval_task.compute_metrics()

    # Add backward-compatible aliases for metric names
    # (ClassificationEvalTask uses acc1/acc5, old validate() used top1/top5)
    if 'acc1' in metrics and 'top1' not in metrics:
        metrics['top1'] = metrics['acc1']
    if 'acc5' in metrics and 'top5' not in metrics:
        metrics['top5'] = metrics['acc5']

    return metrics


def _gather_features(
    features: torch.Tensor,
    device_env: DeviceEnv,
) -> torch.Tensor:
    """Gather features from all distributed ranks.

    Args:
        features: Local features tensor [N_local, D]
        device_env: Device environment with distributed info

    Returns:
        Gathered features [N_total, D] on all ranks
    """
    if not device_env.distributed:
        return features

    # Move to device for all_gather
    features = features.to(device_env.device)

    # Get world size
    world_size = device_env.world_size

    # Gather sizes from all ranks (features may have different counts per rank)
    local_size = torch.tensor([features.size(0)], device=device_env.device)
    all_sizes = [torch.zeros(1, device=device_env.device, dtype=torch.long) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)

    # Pad features to max size for all_gather
    if features.size(0) < max_size:
        padding = torch.zeros(
            max_size - features.size(0),
            features.size(1),
            device=features.device,
            dtype=features.dtype,
        )
        features_padded = torch.cat([features, padding], dim=0)
    else:
        features_padded = features

    # All gather
    gathered = [torch.zeros_like(features_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, features_padded)

    # Remove padding and concatenate
    gathered_trimmed = [g[:s] for g, s in zip(gathered, all_sizes)]
    return torch.cat(gathered_trimmed, dim=0).cpu()


def validate_knn(
    eval_task: 'EvalTask',
    train_loader: DataLoader,
    val_loader: DataLoader,
    device_env: DeviceEnv,
    cfg: TrainConfig,
    log_suffix: str = '',
) -> Dict[str, float]:
    """KNN evaluation using train set as gallery, val set as queries.

    Two-pass evaluation:
    1. Extract features from training set to build the gallery
    2. Evaluate on validation set using KNN against the gallery

    Works with any EvalTask that has get_features() and set_gallery() methods
    (typically SSLEvalTask).

    Args:
        eval_task: EvalTask with KNN support (from task.get_eval_task())
        train_loader: DataLoader for gallery features (typically training set)
        val_loader: DataLoader for query features (validation set)
        device_env: Device environment
        cfg: Training configuration
        log_suffix: Suffix for log messages

    Returns:
        Dict with 'knn_acc' and other metrics from compute_metrics()

    Example::

        eval_task = task.get_eval_task(use_ema=True)
        metrics = validate_knn(
            eval_task, train_loader, val_loader, device_env, cfg
        )
        print(f"KNN accuracy: {metrics['knn_acc']:.2f}%")
    """
    device = device_env.device
    model_dtype = device_env.model_dtype
    amp_autocast = device_env.amp_autocast

    # === Pass 1: Extract gallery features from training set ===
    if is_primary(device_env):
        _logger.info(f'Extracting gallery features{log_suffix}...')

    eval_task.reset()

    with torch.inference_mode():
        for input, target in train_loader:
            if not cfg.loader.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)

            if cfg.device.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                eval_task(input, target)

    gallery_features, gallery_targets = eval_task.get_features()

    # Gather features from all distributed ranks
    if device_env.distributed:
        gallery_features = _gather_features(gallery_features, device_env)
        gallery_targets = _gather_features(gallery_targets.unsqueeze(1), device_env).squeeze(1).long()

    if is_primary(device_env):
        _logger.info(f'Gallery: {gallery_features.size(0)} samples{log_suffix}')

    # === Pass 2: Evaluate on validation set with gallery ===
    if is_primary(device_env):
        _logger.info(f'Evaluating KNN{log_suffix}...')

    eval_task.reset()
    eval_task.set_gallery(gallery_features, gallery_targets)

    with torch.inference_mode():
        for input, target in val_loader:
            if not cfg.loader.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)

            if cfg.device.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                eval_task(input, target)

    metrics = eval_task.compute_metrics()

    if is_primary(device_env):
        _logger.info(f"KNN accuracy{log_suffix}: {metrics.get('knn_acc', 0):.2f}%")

    return metrics
