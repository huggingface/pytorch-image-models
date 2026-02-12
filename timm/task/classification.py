"""Classification training task."""
import logging
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from .task import TrainingTask
from .eval_task import ClassificationEvalTask

_logger = logging.getLogger(__name__)


class ClassificationTask(TrainingTask):
    """Standard supervised classification task.

    Simple task that performs a forward pass through the model and computes
    the classification loss.

    For classification, the model IS the trainable_module (no separate wrapper needed).

    Args:
        model: The model to train
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device for task tensors/buffers
        dtype: Dtype for task tensors/buffers
        verbose: Enable info logging

    Example:
        >>> task = ClassificationTask(model, nn.CrossEntropyLoss(), device=torch.device('cuda'))
        >>> result = task(input, target)
        >>> result['loss'].backward()
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: Union[nn.Module, Callable],
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)
        self.trainable_module = model  # Model IS the trainable module
        self.criterion = criterion

        if self.verbose:
            loss_name = getattr(criterion, '__name__', None) or type(criterion).__name__
            _logger.info(f"ClassificationTask: criterion={loss_name}")

    def get_eval_task(self, use_ema: bool = True) -> ClassificationEvalTask:
        """Get evaluation task for classification.

        Args:
            use_ema: If True and EMA exists, use EMA weights for evaluation

        Returns:
            ClassificationEvalTask configured for this model
        """
        return ClassificationEvalTask(self.get_trainable_module(use_ema))

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
