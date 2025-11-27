"""Classification training task."""
import logging
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from .task import TrainingTask

_logger = logging.getLogger(__name__)


class ClassificationTask(TrainingTask):
    """Standard supervised classification task.

    Simple task that performs a forward pass through the model and computes
    the classification loss.

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
        self.model = model
        self.criterion = criterion

        if self.verbose:
            loss_name = getattr(criterion, '__name__', None) or type(criterion).__name__
            _logger.info(f"ClassificationTask: criterion={loss_name}")

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
        self.model = DDP(self.model, device_ids=device_ids, **ddp_kwargs)
        return self

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
        output = self.model(input)
        loss = self.criterion(output, target)

        return {
            'loss': loss,
            'output': output,
        }
