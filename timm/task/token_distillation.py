"""Token-based distillation training task for models with distillation heads."""
import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from timm.utils import unwrap_model

from .task import TrainingTask

_logger = logging.getLogger(__name__)


class TokenDistillationTeacher(nn.Module):
    """Wrapper for a teacher model used in token-based distillation.

    Creates and manages a pre-trained teacher model for token distillation,
    handling model creation and normalization differences between teacher and student.

    Can be created from:
    - A model name string (creates the model internally)
    - An existing nn.Module (wraps it with the necessary interface)

    Args:
        model_name_or_module: Either a model name string or an nn.Module
        num_classes: Number of output classes (required if model_name_or_module is a string)
        in_chans: Number of input channels (used if model_name_or_module is a string)
        pretrained_path: Optional path to pretrained weights (used if model_name_or_module is a string)
        device: Device to place the model on
        dtype: Model dtype (uses float32 if None)
    """

    def __init__(
            self,
            model_name_or_module: Union[str, nn.Module],
            num_classes: Optional[int] = None,
            in_chans: int = 3,
            pretrained_path: Optional[str] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if isinstance(model_name_or_module, str):
            _logger.info(f"Creating token distillation teacher model: '{model_name_or_module}'")

            pretrained_kwargs = {'pretrained': True}
            if pretrained_path:
                pretrained_kwargs['pretrained_cfg_overlay'] = dict(
                    file=pretrained_path,
                    num_classes=num_classes,
                )

            model = create_model(
                model_name=model_name_or_module,
                num_classes=num_classes,
                in_chans=in_chans,
                device=device,
                dtype=dtype,
                **pretrained_kwargs,
            )
        elif isinstance(model_name_or_module, nn.Module):
            model = model_name_or_module
        else:
            raise TypeError(
                f"model_name_or_module must be a string or nn.Module, got {type(model_name_or_module).__name__}"
            )

        model.eval()
        self.model = model

        # Get normalization values from pretrained_cfg if available
        model_unwrapped = unwrap_model(model)
        if hasattr(model_unwrapped, 'pretrained_cfg'):
            mean = model_unwrapped.pretrained_cfg.get('mean', (0.485, 0.456, 0.406))
            std = model_unwrapped.pretrained_cfg.get('std', (0.229, 0.224, 0.225))
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        mean_kd = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        std_kd = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
        self.register_buffer('mean_kd', mean_kd, persistent=False)
        self.register_buffer('std_kd', std_kd, persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher model.

        Args:
            input: Input tensor (should already be normalized for teacher)

        Returns:
            Teacher logits
        """
        return self.model(input)

    def normalize_input(
            self,
            input: torch.Tensor,
            student_mean: Optional[torch.Tensor] = None,
            student_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize input to match teacher's expected normalization.

        Args:
            input: Input tensor (already normalized for student)
            student_mean: Student normalization mean buffer [1, 3, 1, 1]
            student_std: Student normalization std buffer [1, 3, 1, 1]

        Returns:
            Input tensor normalized for the teacher model
        """
        if student_mean is None or student_std is None:
            return input
        if torch.equal(student_mean, self.mean_kd) and torch.equal(student_std, self.std_kd):
            return input
        return (input * student_std + student_mean - self.mean_kd) / self.std_kd


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
        teacher_model: Teacher model - can be a model name string, nn.Module, or TokenDistillationTeacher
        criterion: Task loss function for main head (default: CrossEntropyLoss)
        teacher_pretrained_path: Path to teacher pretrained weights (used when teacher_model is a string)
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
            teacher_model: Union[str, nn.Module, TokenDistillationTeacher],
            criterion: Optional[nn.Module] = None,
            teacher_pretrained_path: Optional[str] = None,
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

        # Handle different teacher input types
        if isinstance(teacher_model, TokenDistillationTeacher):
            teacher = teacher_model
        elif isinstance(teacher_model, str) or isinstance(teacher_model, nn.Module):
            # Get num_classes and in_chans from student
            num_classes = student_unwrapped.num_classes
            in_chans = student_unwrapped.in_chans
            teacher = TokenDistillationTeacher(
                model_name_or_module=teacher_model,
                num_classes=num_classes,
                in_chans=in_chans,
                pretrained_path=teacher_pretrained_path,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise TypeError(
                f"teacher_model must be a model name string, nn.Module, or TokenDistillationTeacher, "
                f"got {type(teacher_model).__name__}"
            )

        self.student = student_model
        self.teacher = teacher
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
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

        self.student = DDP(self.student, device_ids=device_ids, **ddp_kwargs)
        return self

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
        student_output = self.student(input)
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
