"""Base training task abstraction.

This module provides the base TrainingTask class that encapsulates a complete
forward pass including loss computation. Tasks return a dictionary with loss
components and outputs for logging.
"""
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from timm.utils import unwrap_model

if TYPE_CHECKING:
    from .eval_task import EvalTask


class TrainingTask(nn.Module):
    """Base class for training tasks.

    A training task encapsulates a complete forward pass including loss computation.
    Tasks return a dictionary containing the training loss and other components for logging.

    All tasks use consistent attributes:
        - trainable_module: The module wrapped by DDP, contains all trainable params
        - trainable_module_ema: EMA copy of trainable_module (None if disabled)

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
        >>> # Setup EMA before DDP
        >>> if use_ema:
        >>>     task.setup_ema(decay=0.9999)
        >>>
        >>> # Prepare for distributed training
        >>> if distributed:
        >>>     task.prepare_distributed(device_ids=[local_rank])
        >>>
        >>> # Training loop
        >>> for batch in loader:
        >>>     result = task(input, target)
        >>>     result['loss'].backward()
        >>>     optimizer.step()
        >>>     task.update_ema(step=num_updates)
    """

    # Subclasses should set trainable_module in __init__
    # Type hints only - actual values set in __init__
    trainable_module: nn.Module
    trainable_module_ema: Optional[nn.Module]

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
        self.trainable_module_ema = None
        self._ema_decay = None
        self._ema_device = None

    def to(self, *args, **kwargs):
        """Move task to device/dtype, keeping self.device and self.dtype in sync."""
        dummy = torch.empty(0).to(*args, **kwargs)
        self.device = dummy.device
        self.dtype = dummy.dtype
        return super().to(*args, **kwargs)

    # === Trainable module access ===

    def get_trainable_module(self, use_ema: bool = True) -> nn.Module:
        """Get trainable module, optionally using EMA weights.

        Args:
            use_ema: If True and EMA exists, return EMA module. Otherwise return
                     unwrapped trainable_module.

        Returns:
            The trainable module (unwrapped from DDP if needed)
        """
        if use_ema and self.trainable_module_ema is not None:
            return self.trainable_module_ema
        return unwrap_model(self.trainable_module)

    # === EMA (separate from DDP) ===

    def setup_ema(
            self,
            decay: float = 0.9999,
            warmup: bool = False,
            device: Optional[torch.device] = None,
    ) -> 'TrainingTask':
        """Setup EMA for trainable_module. Call BEFORE prepare_distributed().

        Creates an EMA copy of trainable_module that will be updated after each
        optimizer step via update_ema().

        Args:
            decay: EMA decay rate (default: 0.9999)
            warmup: Whether to use warmup for EMA decay
            device: Device for EMA module (defaults to same as trainable_module)

        Returns:
            self (for method chaining)

        Example:
            >>> task.setup_ema(decay=0.9999).prepare_distributed(device_ids=[rank])
        """
        from timm.utils import ModelEmaV3

        ema_wrapper = ModelEmaV3(
            self.trainable_module,
            decay=decay,
            use_warmup=warmup,
            device=device,
        )
        # Store just the module, not the wrapper
        self.trainable_module_ema = ema_wrapper.module
        self._ema_decay = decay
        self._ema_device = device
        self._ema_wrapper = ema_wrapper  # Keep wrapper for update logic
        return self

    @property
    def has_ema(self) -> bool:
        """Check if EMA is enabled."""
        return self.trainable_module_ema is not None

    def update_ema(self, step: Optional[int] = None) -> None:
        """Update EMA weights. Call after optimizer.step().

        Args:
            step: Current training step (used for warmup if enabled)
        """
        if self._ema_wrapper is not None:
            self._ema_wrapper.update(self.trainable_module, step=step)

    # === DDP ===

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs,
    ) -> 'TrainingTask':
        """Prepare task for distributed training.

        Wraps trainable_module in DistributedDataParallel (DDP) while leaving
        non-trainable components (like frozen teacher models, EMA) unwrapped.

        Should be called AFTER setup_ema() if using both.

        Args:
            device_ids: List of device IDs for DDP (e.g., [local_rank])
            **ddp_kwargs: Additional arguments passed to DistributedDataParallel

        Returns:
            self (for method chaining)

        Example:
            >>> task.setup_ema(decay=0.9999).prepare_distributed(device_ids=[rank])
        """
        if hasattr(self, 'trainable_module') and self.trainable_module is not None:
            self.trainable_module = DDP(
                self.trainable_module,
                device_ids=device_ids,
                **ddp_kwargs,
            )
        return self

    # === Checkpointing (backward compatible format) ===

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Get state dict for checkpointing.

        Returns dict with backward-compatible keys:
            - 'state_dict': trainable_module weights
            - 'state_dict_ema': trainable_module_ema weights (if EMA enabled)

        This format is compatible with existing timm checkpoint infrastructure.
        """
        state = {
            'state_dict': unwrap_model(self.trainable_module).state_dict(),
        }
        if self.trainable_module_ema is not None:
            state['state_dict_ema'] = self.trainable_module_ema.state_dict()
        return state

    def load_state_dict(
            self,
            state_dict: Dict[str, Any],
            strict: bool = True,
    ) -> None:
        """Load state dict, including EMA if present.

        Args:
            state_dict: Dict with 'state_dict' and optionally 'state_dict_ema' keys
            strict: Whether to strictly enforce key matching
        """
        unwrap_model(self.trainable_module).load_state_dict(
            state_dict['state_dict'],
            strict=strict,
        )
        if self.trainable_module_ema is not None and 'state_dict_ema' in state_dict:
            self.trainable_module_ema.load_state_dict(
                state_dict['state_dict_ema'],
                strict=strict,
            )

    def state_dict_for_inference(self) -> Dict[str, torch.Tensor]:
        """Get just core model weights for inference deployment.

        Returns state dict of the core model (.model attribute of trainable_module,
        or trainable_module itself if no .model attribute).

        Override in subclasses if different behavior is needed.
        """
        tm = unwrap_model(self.trainable_module)
        model = getattr(tm, 'model', tm)
        return model.state_dict()

    # === Evaluation ===

    def get_eval_task(self, use_ema: bool = True) -> 'EvalTask':
        """Get evaluation task with current weights.

        Args:
            use_ema: If True and EMA exists, use EMA weights for evaluation

        Returns:
            EvalTask instance configured for this task type

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement get_eval_task()")

    # === Training forward ===

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
