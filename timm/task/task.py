"""Base training task abstraction.

This module provides the base TrainingTask class that encapsulates a complete
forward pass including loss computation. Tasks return a dictionary with loss
components and outputs for logging.
"""
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from timm.utils.model import get_state_dict, unwrap_model
from timm.utils.model_ema import ModelEmaV3


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
            >>> task.compile()
            >>> task.prepare_distributed(device_ids=[args.local_rank])
        """
        # Default implementation - subclasses override if they need DDP
        return self

    def compile(
            self,
            backend: str = 'inductor',
            mode: Optional[str] = None,
            **compile_kwargs,
    ) -> Optional[nn.Module]:
        """Compile hot task components before distributed wrapping.

        Subclasses should compile the train/eval modules that do the tensor
        work, not the outer task wrapper. The return value is the eval-facing
        compiled module/callable used by validation and checkpoint export.
        """
        return None

    def get_trainable_module(self, ema: bool = False) -> Optional[nn.Module]:
        """Return the module that owns trainable parameters."""
        if ema:
            return getattr(self, 'trainable_module_ema', None)
        return getattr(self, 'trainable_module', self)

    def setup_ema(
            self,
            decay: float = 0.9999,
            use_warmup: bool = False,
            device: Optional[torch.device] = None,
            **kwargs,
    ) -> nn.Module:
        """Create an EMA copy of the trainable module."""
        self.trainable_module_ema = ModelEmaV3(
            self.get_trainable_module(),
            decay=decay,
            use_warmup=use_warmup,
            device=device,
            **kwargs,
        )
        return self.trainable_module_ema

    def has_ema(self) -> bool:
        """Return whether this task has an EMA trainable module."""
        return self.get_trainable_module(ema=True) is not None

    def compile_ema(
            self,
            backend: str = 'inductor',
            mode: Optional[str] = None,
            **compile_kwargs,
    ) -> Optional[nn.Module]:
        """Compile the EMA eval model if one has been created."""
        if not self.has_ema():
            return None
        self.eval_model_ema = torch.compile(
            self.get_eval_model(ema=True),
            backend=backend,
            mode=mode,
            **compile_kwargs,
        )
        return self.eval_model_ema

    def update_ema(self, step: Optional[int] = None) -> None:
        """Update EMA state from the current trainable module."""
        if self.has_ema():
            self.get_trainable_module(ema=True).update(unwrap_model(self.get_trainable_module()), step=step)

    def get_clip_parameters(self, exclude_head: bool = False):
        """Return parameters to use for gradient clipping."""
        trainable_module = self.get_trainable_module()
        if exclude_head:
            return [p for p in trainable_module.parameters()][:-2]
        return trainable_module.parameters()

    def get_eval_model(self, module: Optional[nn.Module] = None, ema: bool = False) -> Optional[nn.Module]:
        """Return the eval model/callable used for validation.

        Checkpoint state_dict handling uses unwrap_model separately so DDP and
        compiled wrappers do not leak into saved keys.
        """
        if module is None:
            eval_attr = 'eval_model_ema' if ema else 'eval_model'
            if hasattr(self, eval_attr):
                return getattr(self, eval_attr)
            module = self.get_trainable_module(ema=ema)
        return module

    def get_task_state(self, module: Optional[nn.Module] = None, ema: bool = False) -> Dict[str, Any]:
        """Return task-owned state outside the eval model state_dict."""
        return {}

    def load_task_state(
            self,
            state: Optional[Dict[str, Any]],
            strict: bool = True,
            module: Optional[nn.Module] = None,
            ema: bool = False,
    ) -> None:
        """Load task-owned state outside the eval model state_dict."""
        return None

    def get_checkpoint_state(
            self,
            ema: bool = False,
            unwrap_fn=unwrap_model,
    ) -> Dict[str, Any]:
        """Return checkpoint state entries owned by this task."""
        eval_model = self.get_eval_model(ema=ema)
        if eval_model is None:
            return {}

        model_key = 'state_dict_ema' if ema else 'state_dict'
        task_key = 'task_state_ema' if ema else 'task_state'
        task_state = self.get_task_state(ema=ema)

        state = {model_key: get_state_dict(eval_model, unwrap_fn)}
        if task_state:
            state[task_key] = task_state
        return state

    def load_checkpoint_state(
            self,
            state_dict: Dict[str, Any],
            task_state: Optional[Dict[str, Any]] = None,
            ema: bool = False,
            strict: bool = True,
    ) -> None:
        """Load model and task-owned checkpoint entries."""
        eval_model = self.get_eval_model(ema=ema)
        if eval_model is None:
            if ema:
                raise RuntimeError("Cannot load EMA checkpoint state before setup_ema().")
            raise RuntimeError("Cannot load checkpoint state without an eval model.")
        unwrap_model(eval_model).load_state_dict(state_dict, strict=strict)
        self.load_task_state(task_state, strict=strict, ema=ema)

    def no_sync(self):
        """Return a no-sync context for gradient accumulation.

        Tasks that wrap a trainable component with DDP delegate to that
        component's no_sync(). Non-distributed tasks use a no-op context.
        """
        module = self.get_trainable_module()
        if module is not None and module is not self and hasattr(module, 'no_sync'):
            return module.no_sync()
        return nullcontext()

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
