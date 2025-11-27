"""Base training task abstraction.

This module provides the base TrainingTask class that encapsulates a complete
forward pass including loss computation. Tasks return a dictionary with loss
components and outputs for logging.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn


class TrainingTask(nn.Module):
    """Base class for training tasks.

    A training task encapsulates a complete forward pass including loss computation.
    Tasks return a dictionary containing the training loss and other components for logging.

    The returned dictionary must contain:
        - 'loss': The training loss for backward pass (required)
        - 'output': Model output/logits for metric computation (recommended)
        - Other task-specific loss components for logging (optional)

    Args:
        device: Device for task tensors/buffers (defaults to cpu)
        dtype: Dtype for task tensors/buffers (defaults to torch default)
        verbose: Enable info logging

    Example:
        >>> task = SomeTask(model, criterion, device=torch.device('cuda'))
        >>>
        >>> # Prepare for distributed training (if needed)
        >>> if distributed:
        >>>     task.prepare_distributed(device_ids=[local_rank])
        >>>
        >>> # Training loop
        >>> result = task(input, target)
        >>> result['loss'].backward()
    """

    def __init__(
            self,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.verbose = verbose

    def to(self, *args, **kwargs):
        """Move task to device/dtype, keeping self.device and self.dtype in sync."""
        dummy = torch.empty(0).to(*args, **kwargs)
        self.device = dummy.device
        self.dtype = dummy.dtype
        return super().to(*args, **kwargs)

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs
    ) -> 'TrainingTask':
        """Prepare task for distributed training.

        This method wraps trainable components in DistributedDataParallel (DDP)
        while leaving non-trainable components (like frozen teacher models) unwrapped.

        Should be called after task initialization but before training loop.

        Args:
            device_ids: List of device IDs for DDP (e.g., [local_rank])
            **ddp_kwargs: Additional arguments passed to DistributedDataParallel

        Returns:
            self (for method chaining)

        Example:
            >>> task = LogitDistillationTask(student, teacher, criterion)
            >>> task.prepare_distributed(device_ids=[args.local_rank])
            >>> task = torch.compile(task)  # Compile after DDP
        """
        # Default implementation - subclasses override if they need DDP
        return self

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform forward pass and compute loss.

        Args:
            input: Input tensor [B, C, H, W]
            target: Target labels [B]

        Returns:
            Dictionary with at least 'loss' key containing the training loss
        """
        raise NotImplementedError
