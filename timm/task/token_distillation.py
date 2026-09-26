"""Token-based distillation training task for models with distillation heads."""
import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import unwrap_model

from .classification import resolve_classification_loss
from .distillation import DistillationTeacher, _resolve_teacher
from .task import TrainingTask

_logger = logging.getLogger(__name__)


class TokenDistillationTask(TrainingTask):
    """Token-based distillation task for models with distillation heads.

    For models like DeiT that have a dedicated distillation token/head that returns
    a tuple (main_logits, dist_logits) when distilled_training is enabled. The main
    head is trained against ground truth labels while the distillation head matches
    teacher outputs.

    Supports two distillation modes:
    - 'soft': KL divergence with temperature scaling (default)
    - 'hard': Cross-entropy with teacher's hard predictions (argmax)

    Loss weighting supports two modes:
    1. Independent weights: loss = task_loss_weight * task_loss + distill_loss_weight * distill_loss
    2. Complementary mode: loss = task_loss_weight * task_loss + (1 - task_loss_weight) * distill_loss
       (used when only task_loss_weight is specified)

    Args:
        student_model: Student model with set_distilled_training() method
        teacher_model: Teacher model - can be a model name string, nn.Module, or DistillationTeacher
        criterion: Task loss function for main head. Created by create_classification_loss() from
            criterion_kwargs when None.
        criterion_kwargs: Arguments for create_classification_loss() when criterion is None.
        teacher_pretrained_path: Path to teacher pretrained weights (used when teacher_model is a string)
        teacher_pretrained_cfg_overlay: Teacher pretrained cfg overrides (used when teacher_model is a string)
        distill_type: 'soft' for KL-div or 'hard' for CE with teacher argmax
        distill_loss_weight: Weight for distillation loss
        task_loss_weight: Weight for task loss
        temperature: Softmax temperature for soft distillation (ignored for hard)
        device: Device for task tensors/buffers
        dtype: Dtype for task tensors/buffers
        verbose: Enable info logging

    Example:
        >>> # With model name string (num_classes/in_chans inferred from student)
        >>> task = TokenDistillationTask(
        ...     student_model=model, teacher_model='deit_base_patch16_224',
        ...     criterion=nn.CrossEntropyLoss(),
        ...     distill_type='soft', temperature=3.0, task_loss_weight=0.5,
        ...     device=torch.device('cuda'),
        ... )
        >>> # With raw model
        >>> task = TokenDistillationTask(
        ...     student_model=model, teacher_model=my_teacher_model,
        ...     criterion=nn.CrossEntropyLoss(),
        ...     distill_type='hard', task_loss_weight=0.5,
        ... )
    """

    def __init__(
            self,
            student_model: nn.Module,
            teacher_model: Union[str, nn.Module, DistillationTeacher],
            criterion: Optional[nn.Module] = None,
            criterion_kwargs: Optional[Dict[str, Any]] = None,
            teacher_pretrained_path: Optional[str] = None,
            teacher_pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
            distill_type: str = 'soft',
            distill_loss_weight: Optional[float] = None,
            task_loss_weight: Optional[float] = None,
            temperature: float = 1.0,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)

        # Validate model has set_distilled_training method
        student_unwrapped = unwrap_model(student_model)
        if not hasattr(student_unwrapped, 'set_distilled_training'):
            raise ValueError(
                f"Model {student_unwrapped.__class__.__name__} does not have 'set_distilled_training' method. "
                "TokenDistillationTask requires a model with a distillation head (e.g., DeiT distilled variants)."
            )

        # Enable distilled training mode
        student_unwrapped.set_distilled_training(True)

        self.trainable_module = student_model
        self.teacher = _resolve_teacher(
            teacher=teacher_model,
            student_model=student_model,
            pretrained_path=teacher_pretrained_path,
            pretrained_cfg_overlay=teacher_pretrained_cfg_overlay,
            device=self.device,
            dtype=self.dtype,
        )
        self.criterion = resolve_classification_loss(criterion, self.device, criterion_kwargs=criterion_kwargs)
        self.distill_type = distill_type
        self.temperature = temperature

        if distill_type not in ('soft', 'hard'):
            raise ValueError(f"Unsupported distill_type '{distill_type}'. Must be 'soft' or 'hard'.")

        # Register student normalization values as non-persistent buffers
        student_mean = torch.tensor(
            student_unwrapped.pretrained_cfg['mean'],
            device=self.device,
            dtype=self.dtype,
        ).view(1, -1, 1, 1)
        student_std = torch.tensor(
            student_unwrapped.pretrained_cfg['std'],
            device=self.device,
            dtype=self.dtype,
        ).view(1, -1, 1, 1)
        self.register_buffer('student_mean', student_mean, persistent=False)
        self.register_buffer('student_std', student_std, persistent=False)

        # Determine weighting mode
        if distill_loss_weight is not None:
            # Mode 1: distill_weight specified - independent weights (task defaults to 1.0 if not set)
            self.distill_loss_weight = distill_loss_weight
            self.task_loss_weight = task_loss_weight if task_loss_weight is not None else 1.0
            if self.verbose:
                _logger.info(
                    f"TokenDistillationTask: Independent weights - "
                    f"task_weight={self.task_loss_weight}, distill_weight={distill_loss_weight}"
                )
        elif task_loss_weight is not None:
            # Mode 2: only task_weight specified - complementary mode (distill = 1 - task)
            self.task_loss_weight = task_loss_weight
            self.distill_loss_weight = 1.0 - task_loss_weight
            if self.verbose:
                _logger.info(
                    f"TokenDistillationTask: Complementary mode - "
                    f"task_weight={task_loss_weight}, distill_weight={self.distill_loss_weight}"
                )
        else:
            # Mode 3: neither specified - equal weights (both 1.0)
            self.distill_loss_weight = 1.0
            self.task_loss_weight = 1.0
            if self.verbose:
                _logger.info(
                    f"TokenDistillationTask: Default equal weights - "
                    f"task_weight={self.task_loss_weight}, distill_weight={self.distill_loss_weight}"
                )

        if self.verbose:
            _logger.info(
                f"TokenDistillationTask: distill_type={distill_type}, temperature={temperature}"
            )

    def train(self, mode: bool = True) -> 'TokenDistillationTask':
        """Set task training mode while keeping the teacher in evaluation mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs
    ) -> 'TokenDistillationTask':
        """Prepare task for distributed training.

        Wraps the student model in DistributedDataParallel (DDP) while leaving
        the frozen teacher model unwrapped.

        Args:
            device_ids: List of device IDs for DDP (e.g., [local_rank])
            **ddp_kwargs: Additional arguments passed to DistributedDataParallel

        Returns:
            self (for method chaining)
        """
        from torch.nn.parallel import DistributedDataParallel as DDP

        for param in self.teacher.parameters():
            param.requires_grad = False

        self.trainable_module = DDP(self.trainable_module, device_ids=device_ids, **ddp_kwargs)
        return self

    def compile(
            self,
            backend: str = 'inductor',
            mode: Optional[str] = None,
            **compile_kwargs,
    ) -> nn.Module:
        """Compile student eval/train forward and teacher logit forward."""
        self.trainable_module = torch.compile(self.trainable_module, backend=backend, mode=mode, **compile_kwargs)
        # Token distillation keeps the student as the eval-facing model.
        self.eval_model = self.trainable_module
        self.teacher.compile(backend=backend, mode=mode, **compile_kwargs)
        return self.trainable_module

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with token distillation.

        Args:
            input: Input tensor [B, C, H, W]
            target: Target labels [B]

        Returns:
            Dictionary containing:
                - 'loss': Combined training loss (task + distillation)
                - 'output': Main head logits (for metrics)
                - 'task_loss': Classification loss component
                - 'distill_loss': Distillation loss component
        """
        # Student forward pass - returns tuple (main_logits, dist_logits)
        student_output = self.trainable_module(input)
        main_logits, dist_logits = student_output

        # Compute task loss on main head
        task_loss = self.criterion(main_logits, target)

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            input_kd = self.teacher.normalize_input(input, self.student_mean, self.student_std)
            teacher_logits = self.teacher(input_kd.detach())

        # Compute distillation loss on distillation head
        if self.distill_type == 'soft':
            prob_s = F.log_softmax(dist_logits / self.temperature, dim=-1)
            prob_t = F.log_softmax(teacher_logits / self.temperature, dim=-1)
            distill_loss = F.kl_div(prob_s, prob_t, reduction='batchmean', log_target=True) * (self.temperature ** 2)
        else:
            teacher_hard = teacher_logits.argmax(dim=-1)
            distill_loss = F.cross_entropy(dist_logits, teacher_hard)

        total_loss = self.task_loss_weight * task_loss + self.distill_loss_weight * distill_loss

        return {
            'loss': total_loss,
            'output': main_logits,
            'task_loss': task_loss,
            'distill_loss': distill_loss,
        }
