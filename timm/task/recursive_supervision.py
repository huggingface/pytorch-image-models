"""Recursive supervision training task for RSViT models."""
import logging
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .task import TrainingTask

_logger = logging.getLogger(__name__)


class RecursiveSupervisionTask(TrainingTask):
    """Recursive supervision training task for models like RSViT.

    Computes weighted cross-entropy loss at each supervision step during training,
    plus a halting loss that trains the model to predict whether its answer is correct.

    Args:
        model: The TRM model to train.
        criterion: Base loss function (e.g., CrossEntropyLoss).
        n_sup: Number of supervision steps.
        step_weights: Weights for each supervision step. If None, uses linearly
            increasing weights (later steps weighted more heavily).
        halt_weight: Weight for the halting loss (default: 0.1).
        device: Device for task tensors/buffers.
        dtype: Dtype for task tensors/buffers.
        verbose: Enable info logging.

    Example:
        >>> model = create_model('rspvit_tiny_patch16_224')
        >>> task = RecursiveSupervisionTask(model=model, criterion=nn.CrossEntropyLoss(), n_sup=4)
        >>> result = task(input, target)
        >>> result['loss'].backward()
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: Union[nn.Module, Callable],
            n_sup: int = 4,
            step_weights: Optional[List[float]] = None,
            halt_weight: float = 0.1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)
        self.model = model
        self.criterion = criterion
        self.n_sup = n_sup
        self.halt_weight = halt_weight

        # Set up step weights (linearly increasing if not specified)
        if step_weights is None:
            # Linearly increasing: [1, 2, 3, 4] -> normalized
            weights = [(i + 1) for i in range(n_sup)]
            total = sum(weights)
            step_weights = [w / total for w in weights]
        else:
            if len(step_weights) != n_sup:
                raise ValueError(
                    f"step_weights length ({len(step_weights)}) must match n_sup ({n_sup})"
                )
            # Normalize weights
            total = sum(step_weights)
            step_weights = [w / total for w in step_weights]

        self.step_weights = step_weights

        # Binary cross entropy for halting prediction
        self.halt_criterion = nn.BCEWithLogitsLoss()

        if self.verbose:
            loss_name = getattr(criterion, '__name__', None) or type(criterion).__name__
            _logger.info(
                f"RecursiveSupervisionTask: criterion={loss_name}, n_sup={n_sup}, "
                f"step_weights={[f'{w:.3f}' for w in self.step_weights]}, "
                f"halt_weight={halt_weight}"
            )

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs
    ) -> 'RecursiveSupervisionTask':
        """Prepare task for distributed training.

        Wraps the model in DistributedDataParallel (DDP).

        Args:
            device_ids: List of device IDs for DDP (e.g., [local_rank]).
            **ddp_kwargs: Additional arguments passed to DistributedDataParallel.

        Returns:
            self (for method chaining).
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        self.model = DDP(self.model, device_ids=device_ids, **ddp_kwargs)
        return self

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with deep supervision loss computation.

        Args:
            input: Input tensor [B, C, H, W].
            target: Target labels [B].

        Returns:
            Dictionary containing:
                - 'loss': Total training loss (task_loss + halt_weight * halt_loss).
                - 'output': Final step logits for metric computation.
                - 'task_loss': Weighted sum of classification losses.
                - 'halt_loss': Binary cross-entropy halting loss.
                - 'step_losses': List of individual step losses.
        """
        result = self.model(input, return_all_steps=True)
        step_logits = result['step_logits']
        halt_logits = result['halt_logits']

        # Compute weighted task loss
        step_losses = []
        task_loss = 0.0

        for logits, weight in zip(step_logits, self.step_weights):
            step_loss = self.criterion(logits, target)
            step_losses.append(step_loss.detach())
            task_loss = task_loss + weight * step_loss

        # Compute halting loss
        # Handle soft targets (mixup/cutmix): [B, num_classes] -> [B]
        if target.ndim == 2:
            target_labels = target.argmax(dim=-1)
        else:
            target_labels = target

        halt_loss = 0.0
        for logits, halt in zip(step_logits, halt_logits):
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                is_correct = (predictions == target_labels).float().unsqueeze(-1)  # [B, 1]

            step_halt_loss = self.halt_criterion(halt, is_correct)
            halt_loss = halt_loss + step_halt_loss

        halt_loss = halt_loss / len(step_logits)

        # Total loss
        total_loss = task_loss + self.halt_weight * halt_loss

        # Use final step output for metrics
        final_output = step_logits[-1]

        return {
            'loss': total_loss,
            'output': final_output,
            'task_loss': task_loss.detach() if isinstance(task_loss, torch.Tensor) else task_loss,
            'halt_loss': halt_loss.detach() if isinstance(halt_loss, torch.Tensor) else halt_loss,
            'step_losses': step_losses,
        }
