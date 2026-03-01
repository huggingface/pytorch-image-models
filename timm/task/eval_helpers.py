"""Evaluation helper functions.

This module provides helper functions for common evaluation patterns.
"""
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .eval_task import EvalTask


def evaluate(
        eval_task: EvalTask,
        data_loader: DataLoader,
        device: torch.device,
        log_fn: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None,
) -> Dict[str, float]:
    """Generic evaluation loop that works for any EvalTask.

    Runs the eval task on all batches in the data loader, accumulating
    metrics internally, then returns the final computed metrics.

    Args:
        eval_task: EvalTask instance (from task.get_eval_task())
        data_loader: DataLoader for evaluation data
        device: Device for computation
        log_fn: Optional callback for logging running metrics per batch.
                Called with the result dict from eval_task forward().

    Returns:
        Dictionary of final metrics from eval_task.compute_metrics()

    Example:
        >>> eval_task = task.get_eval_task(use_ema=True)
        >>> metrics = evaluate(eval_task, val_loader, device)
        >>> print(f"Accuracy: {metrics['acc1']:.2f}%")
    """
    eval_task.reset()

    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        result = eval_task(input, target)

        if log_fn is not None:
            log_fn(result)

    return eval_task.compute_metrics()


def evaluate_knn(
        eval_task: EvalTask,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        log_fn: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None,
) -> Dict[str, float]:
    """KNN evaluation using train set as gallery, val set as queries.

    Performs a two-pass evaluation:
    1. Extract features from training set to build the gallery
    2. Evaluate on validation set using KNN against the gallery

    Works with any EvalTask that has get_features() and set_gallery() methods
    (typically SSLEvalTask).

    Args:
        eval_task: EvalTask instance with KNN support (from task.get_eval_task())
        train_loader: DataLoader for gallery features (typically training set)
        val_loader: DataLoader for query features (validation set)
        device: Device for computation
        log_fn: Optional callback for logging running metrics per batch.
                Only called during validation pass.

    Returns:
        Dictionary with 'knn_acc' and other metrics from compute_metrics()

    Example:
        >>> eval_task = task.get_eval_task(use_ema=True)
        >>> metrics = evaluate_knn(eval_task, train_loader, val_loader, device)
        >>> print(f"KNN accuracy: {metrics['knn_acc']:.2f}%")
    """
    # Pass 1: Extract gallery features from training set
    eval_task.reset()
    for input, target in train_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        eval_task(input, target)

    gallery_features, gallery_targets = eval_task.get_features()

    # Pass 2: Evaluate on validation set with gallery
    eval_task.reset()
    eval_task.set_gallery(gallery_features, gallery_targets)

    for input, target in val_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        result = eval_task(input, target)

        if log_fn is not None:
            log_fn(result)

    return eval_task.compute_metrics()
