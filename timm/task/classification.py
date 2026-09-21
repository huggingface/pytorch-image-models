"""Classification training task."""
import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from timm.loss import BinaryCrossEntropy, JsdCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from .task import TrainingTask

_logger = logging.getLogger(__name__)


def create_classification_loss(
        multi_label: bool = False,
        bce: bool = False,
        smoothing: float = 0.,
        soft_targets: bool = False,
        jsd_splits: int = 0,
        bce_target_thresh: Optional[float] = None,
        bce_sum: bool = False,
        bce_pos_weight: Optional[Union[torch.Tensor, float]] = None,
) -> nn.Module:
    """Create the training criterion for a classification task.

    Args:
        multi_label: Targets are dense independent binary labels shaped (B, C), always trained with BCE.
        bce: Use binary cross-entropy instead of softmax cross-entropy for single-label targets.
        smoothing: Label smoothing factor. Ignored when soft_targets is set, since Mixup/CutMix
            already apply it while producing the soft targets.
        soft_targets: Targets arrive as dense soft distributions from Mixup/CutMix rather than class indices.
        jsd_splits: Number of augmentation splits for the JSD loss. Values above 1 select JsdCrossEntropy.
        bce_target_thresh: Binarize soft targets above this value (BCE only).
        bce_sum: Sum the loss over classes before averaging over the batch (BCE only).
        bce_pos_weight: Positive class weight (BCE only).
    """
    if jsd_splits > 1:
        if multi_label:
            raise ValueError('JSD loss is not supported for multi-label targets.')
        return JsdCrossEntropy(num_splits=jsd_splits, smoothing=smoothing)
    if multi_label or bce:
        return BinaryCrossEntropy(
            smoothing=0. if soft_targets else smoothing,
            smooth_dense=multi_label,
            target_threshold=bce_target_thresh,
            sum_classes=bce_sum,
            pos_weight=bce_pos_weight,
        )
    if soft_targets:
        return SoftTargetCrossEntropy()
    if smoothing:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    return nn.CrossEntropyLoss()


def resolve_classification_loss(
        criterion: Optional[Union[nn.Module, Callable]],
        device: torch.device,
        multi_label: bool = False,
        criterion_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[nn.Module, Callable]:
    """Return an explicit criterion moved to device, or create one with create_classification_loss(**criterion_kwargs)."""
    if criterion is None:
        criterion = create_classification_loss(multi_label=multi_label, **(criterion_kwargs or {}))
    elif criterion_kwargs:
        raise ValueError(
            f'Pass either an explicit criterion or criterion_kwargs, not both: {sorted(criterion_kwargs)}')
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device=device)
    return criterion


class ClassificationTask(TrainingTask):
    """Standard supervised classification task.

    Simple task that performs a forward pass through the model and computes
    the classification loss. The criterion is created by create_classification_loss()
    unless one is passed explicitly.

    Args:
        model: The model to train
        criterion: Loss function. Created from criterion_kwargs when None.
        criterion_kwargs: Arguments for create_classification_loss() when criterion is None.
        device: Device for task tensors/buffers
        dtype: Dtype for task tensors/buffers
        verbose: Enable info logging

    Example:
        >>> task = ClassificationTask(model, criterion_kwargs=dict(smoothing=0.1), device=torch.device('cuda'))
        >>> result = task(input, target)
        >>> result['loss'].backward()
    """

    multi_label = False

    def __init__(
            self,
            model: nn.Module,
            criterion: Optional[Union[nn.Module, Callable]] = None,
            criterion_kwargs: Optional[Dict[str, Any]] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)
        self.trainable_module = model
        self.criterion = resolve_classification_loss(
            criterion,
            self.device,
            multi_label=self.multi_label,
            criterion_kwargs=criterion_kwargs,
        )

        if self.verbose:
            loss_name = getattr(self.criterion, '__name__', None) or type(self.criterion).__name__
            _logger.info(f"{type(self).__name__}: criterion={loss_name}")

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs
    ) -> 'ClassificationTask':
        """Prepare task for distributed training.

        Wraps the model in DistributedDataParallel (DDP).

        Args:
            device_ids: List of device IDs for DDP (e.g., [local_rank])
            **ddp_kwargs: Additional arguments passed to DistributedDataParallel

        Returns:
            self (for method chaining)
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        self.trainable_module = DDP(self.trainable_module, device_ids=device_ids, **ddp_kwargs)
        return self

    def compile(
            self,
            backend: str = 'inductor',
            mode: Optional[str] = None,
            **compile_kwargs,
    ) -> nn.Module:
        """Compile the classification model before DDP wrapping."""
        self.trainable_module = torch.compile(self.trainable_module, backend=backend, mode=mode, **compile_kwargs)
        # Classification uses the same forward for training and eval.
        self.eval_model = self.trainable_module
        return self.trainable_module

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through model and compute classification loss.

        Args:
            input: Input tensor [B, C, H, W]
            target: Target labels [B]

        Returns:
            Dictionary containing:
                - 'loss': Classification loss
                - 'output': Model logits
        """
        output = self.trainable_module(input)
        loss = self.criterion(output, target)

        return {
            'loss': loss,
            'output': output,
        }


class MultiLabelClassificationTask(ClassificationTask):
    """Independent binary classification with dense float targets shaped (B, C).

    The default criterion is BinaryCrossEntropy with dense-target smoothing. When
    Mixup/CutMix are active they apply the smoothing while producing soft targets
    and the criterion is created with soft_targets=True instead. Evaluation always
    uses unsmoothed binary targets. Model, distributed, compilation, EMA, and
    checkpoint handling are inherited.

    Args:
        model: The model to train
        criterion: Loss function. Created from criterion_kwargs when None.
        criterion_kwargs: Arguments for create_classification_loss() when criterion is None.
        threshold: Sigmoid probability threshold for the evaluator's F1 metrics.
        device: Device for task tensors/buffers
        dtype: Dtype for task tensors/buffers
        verbose: Enable info logging
    """

    multi_label = True

    def __init__(
            self,
            model: nn.Module,
            criterion: Optional[Union[nn.Module, Callable]] = None,
            criterion_kwargs: Optional[Dict[str, Any]] = None,
            threshold: float = 0.5,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        if not 0. < threshold < 1.:
            raise ValueError('Multi-label prediction threshold must be between 0 and 1.')
        super().__init__(
            model, criterion, criterion_kwargs=criterion_kwargs, device=device, dtype=dtype, verbose=verbose)
        self.threshold = threshold

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through model and compute the multi-label loss.

        Args:
            input: Input tensor [B, C, H, W]
            target: Dense float targets [B, num_classes]

        Returns:
            Dictionary containing:
                - 'loss': Binary classification loss
                - 'output': Model logits
        """
        if target.ndim != 2 or not target.is_floating_point():
            raise ValueError('Multi-label training expects dense floating-point targets shaped (batch, num_classes).')
        return super().forward(input, target)

    def create_evaluator(self, **kwargs):
        from .evaluator import MultiLabelClassificationEvaluator
        return MultiLabelClassificationEvaluator(device=self.device, threshold=self.threshold, **kwargs)
