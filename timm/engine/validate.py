"""Validation utilities for timm engine.

Provides the validation loop for evaluating model performance.
"""
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from timm.utils import AverageMeter, accuracy, reduce_tensor

from .config import TrainConfig
from .device import DeviceEnv, is_primary

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
