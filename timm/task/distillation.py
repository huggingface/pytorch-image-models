"""Knowledge distillation training tasks and components."""
import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from timm.utils import unwrap_model

from .task import TrainingTask
from .eval_task import ClassificationEvalTask


_logger = logging.getLogger(__name__)


class DistillationTeacher(nn.Module):
    """Wrapper for a teacher model used in knowledge distillation.

    Creates and manages a pre-trained teacher model for knowledge distillation,
    handling model creation and normalization differences between teacher and student.

    Can be created from:
    - A model name string (creates the model internally with pretrained weights)
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
            _logger.info(f"Creating KD teacher model: '{model_name_or_module}'")

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

    def forward(
            self,
            input: torch.Tensor,
            return_features: bool = False,
    ) -> torch.Tensor:
        """Forward pass through teacher model.

        Args:
            input: Input tensor (should already be normalized for teacher)
            return_features: Whether to return pooled pre-logits features instead of logits

        Returns:
            Logits or pooled pre-logits features depending on return_features flag
        """
        if return_features:
            if not hasattr(self.model, 'forward_features') or not hasattr(self.model, 'forward_head'):
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not support feature extraction. "
                    "Ensure the model has 'forward_features' and 'forward_head' methods."
                )
            feature_map = self.model.forward_features(input)
            return self.model.forward_head(feature_map, pre_logits=True)
        else:
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


def _resolve_teacher(
        teacher: Union[str, nn.Module, DistillationTeacher],
        student_model: nn.Module,
        pretrained_path: Optional[str],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
) -> DistillationTeacher:
    """Resolve teacher input to a DistillationTeacher instance.

    Args:
        teacher: Model name string, nn.Module, or DistillationTeacher
        student_model: Student model to infer num_classes/in_chans from
        pretrained_path: Optional path to teacher pretrained weights
        device: Device for teacher
        dtype: Dtype for teacher

    Returns:
        DistillationTeacher instance
    """
    if isinstance(teacher, DistillationTeacher):
        return teacher

    # Get num_classes and in_chans from student
    student_unwrapped = unwrap_model(student_model)
    num_classes = student_unwrapped.num_classes
    in_chans = student_unwrapped.in_chans

    return DistillationTeacher(
        model_name_or_module=teacher,
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained_path=pretrained_path,
        device=device,
        dtype=dtype,
    )


class LogitDistillationTask(TrainingTask):
    """Logit-based knowledge distillation task.

    Performs distillation by matching student and teacher output logits using
    KL divergence with temperature scaling.

    Loss weighting supports two modes:
    1. Independent weights: loss = task_loss_weight * task_loss + distill_loss_weight * distill_loss
    2. Complementary mode: loss = task_loss_weight * task_loss + (1 - task_loss_weight) * distill_loss
       (used when only task_loss_weight is specified)

    Args:
        student_model: Student model to train
        teacher_model: Teacher model - can be a model name string, nn.Module, or DistillationTeacher
        criterion: Task loss function (default: CrossEntropyLoss)
        teacher_pretrained_path: Path to teacher pretrained weights (used when teacher_model is a string)
        loss_type: Type of distillation loss (currently only 'kl' supported)
        distill_loss_weight: Weight for distillation loss
        task_loss_weight: Weight for task loss
        temperature: Softmax temperature for distillation (typical values: 1-4)
        device: Device for task tensors/buffers
        dtype: Dtype for task tensors/buffers
        verbose: Enable info logging

    Example:
        >>> # With model name string (num_classes/in_chans inferred from student)
        >>> task = LogitDistillationTask(
        ...     student_model=model, teacher_model='resnet50',
        ...     criterion=nn.CrossEntropyLoss(),
        ...     task_loss_weight=0.3, temperature=4.0,
        ...     device=torch.device('cuda'),
        ... )
        >>> # With raw model
        >>> task = LogitDistillationTask(
        ...     student_model=model, teacher_model=my_teacher_model,
        ...     criterion=nn.CrossEntropyLoss(),
        ...     task_loss_weight=0.3, temperature=4.0,
        ... )
    """

    def __init__(
            self,
            student_model: nn.Module,
            teacher_model: Union[str, nn.Module, DistillationTeacher],
            criterion: Optional[nn.Module] = None,
            teacher_pretrained_path: Optional[str] = None,
            loss_type: str = 'kl',
            distill_loss_weight: Optional[float] = None,
            task_loss_weight: Optional[float] = None,
            temperature: float = 1.0,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)

        # Resolve teacher to DistillationTeacher
        teacher = _resolve_teacher(
            teacher_model,
            student_model,
            teacher_pretrained_path,
            self.device,
            self.dtype,
        )

        self.trainable_module = student_model  # Student IS the trainable module
        self.teacher = teacher
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.loss_type = loss_type
        self.temperature = temperature

        if loss_type != 'kl':
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Currently only 'kl' is supported.")

        # Register student normalization values as non-persistent buffers
        student_unwrapped = unwrap_model(student_model)
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
                    f"LogitDistillationTask: Independent weights - "
                    f"task_weight={self.task_loss_weight}, distill_weight={distill_loss_weight}"
                )
        elif task_loss_weight is not None:
            # Mode 2: only task_weight specified - complementary mode (distill = 1 - task)
            self.task_loss_weight = task_loss_weight
            self.distill_loss_weight = 1.0 - task_loss_weight
            if self.verbose:
                _logger.info(
                    f"LogitDistillationTask: Complementary mode - "
                    f"task_weight={task_loss_weight}, distill_weight={self.distill_loss_weight}"
                )
        else:
            # Mode 3: neither specified - equal weights (both 1.0)
            self.distill_loss_weight = 1.0
            self.task_loss_weight = 1.0
            if self.verbose:
                _logger.info(
                    f"LogitDistillationTask: Default equal weights - "
                    f"task_weight={self.task_loss_weight}, distill_weight={self.distill_loss_weight}"
                )

        if self.verbose:
            _logger.info(
                f"LogitDistillationTask: loss_type={loss_type}, temperature={temperature}"
            )

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs
    ) -> 'LogitDistillationTask':
        """Prepare task for distributed training.

        Wraps the student model in DistributedDataParallel (DDP) while leaving
        the frozen teacher model unwrapped.

        Args:
            device_ids: List of device IDs for DDP (e.g., [local_rank])
            **ddp_kwargs: Additional arguments passed to DistributedDataParallel

        Returns:
            self (for method chaining)
        """
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Use base class for DDP wrapping of trainable_module
        return super().prepare_distributed(device_ids=device_ids, **ddp_kwargs)

    def get_eval_task(self, use_ema: bool = True) -> ClassificationEvalTask:
        """Get evaluation task for classification.

        Args:
            use_ema: If True and EMA exists, use EMA weights for evaluation

        Returns:
            ClassificationEvalTask configured for the student model
        """
        return ClassificationEvalTask(self.get_trainable_module(use_ema))

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with logit distillation.

        Args:
            input: Input tensor [B, C, H, W]
            target: Target labels [B]

        Returns:
            Dictionary containing:
                - 'loss': Combined training loss (task + distillation)
                - 'output': Student logits (for metrics)
                - 'task_loss': Classification loss component
                - 'kd_loss': Logit distillation loss component
        """
        student_logits = self.trainable_module(input)
        task_loss = self.criterion(student_logits, target)

        with torch.no_grad():
            input_kd = self.teacher.normalize_input(input, self.student_mean, self.student_std)
            teacher_logits = self.teacher(input_kd.detach(), return_features=False)

        prob_s = F.log_softmax(student_logits / self.temperature, dim=-1)
        prob_t = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        kd_loss = F.kl_div(prob_s, prob_t, reduction='batchmean', log_target=True) * (self.temperature ** 2)

        total_loss = self.task_loss_weight * task_loss + self.distill_loss_weight * kd_loss

        return {
            'loss': total_loss,
            'output': student_logits,
            'task_loss': task_loss,
            'kd_loss': kd_loss,
        }


class FeatureDistillationTrainableModule(nn.Module):
    """Trainable module for feature distillation.

    Wraps student model and projection layer into a single module where all
    trainable forward operations happen inside forward(). This ensures proper
    DDP wrapping when the module is used with DistributedDataParallel.
    """

    def __init__(
            self,
            model: nn.Module,
            projection: Optional[nn.Module] = None,
    ):
        """ Create trainable module wrapper for feature distillation.

        Args:
            model: Student model to train (core model)
            projection: Optional projection layer (Linear layer or None)
        """
        super().__init__()
        self.model = model  # Core student model
        self.projection = projection

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through student and projection.

        Args:
            input: Input tensor [B, C, H, W]

        Returns:
            Tuple of (student_logits, student_features) where features are
            optionally projected to match teacher dimension.
        """
        feature_map = self.model.forward_features(input)
        student_logits = self.model.forward_head(feature_map)
        student_features = self.model.forward_head(feature_map, pre_logits=True)

        if self.projection is not None:
            student_features = self.projection(student_features)

        return student_logits, student_features


class FeatureDistillationTask(TrainingTask):
    """Feature-based knowledge distillation task.

    Performs distillation by matching student and teacher intermediate features
    (pooled pre-logits) using MSE loss. Automatically creates a projection layer
    if student and teacher feature dimensions differ.

    Loss weighting supports two modes:
    1. Independent weights: loss = task_loss_weight * task_loss + distill_loss_weight * distill_loss
    2. Complementary mode: loss = task_loss_weight * task_loss + (1 - task_loss_weight) * distill_loss
       (used when only task_loss_weight is specified)

    Args:
        student_model: Student model to train
        teacher_model: Teacher model - can be a model name string, nn.Module, or DistillationTeacher
        criterion: Task loss function (default: CrossEntropyLoss)
        teacher_pretrained_path: Path to teacher pretrained weights (used when teacher_model is a string)
        distill_loss_weight: Weight for distillation loss
        task_loss_weight: Weight for task loss
        student_feature_dim: Student pre-logits dimension (auto-detected if None)
        teacher_feature_dim: Teacher pre-logits dimension (auto-detected if None)
        device: Device for task tensors/buffers
        dtype: Dtype for task tensors/buffers
        verbose: Enable info logging

    Example:
        >>> # With model name string (num_classes/in_chans inferred from student)
        >>> task = FeatureDistillationTask(
        ...     student_model=model, teacher_model='resnet50',
        ...     criterion=nn.CrossEntropyLoss(),
        ...     distill_loss_weight=5.0, task_loss_weight=1.0,
        ...     device=torch.device('cuda'),
        ... )
    """

    def __init__(
            self,
            student_model: nn.Module,
            teacher_model: Union[str, nn.Module, DistillationTeacher],
            criterion: Optional[nn.Module] = None,
            teacher_pretrained_path: Optional[str] = None,
            distill_loss_weight: Optional[float] = None,
            task_loss_weight: Optional[float] = None,
            student_feature_dim: Optional[int] = None,
            teacher_feature_dim: Optional[int] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)

        # Resolve teacher to DistillationTeacher
        teacher = _resolve_teacher(
            teacher_model,
            student_model,
            teacher_pretrained_path,
            self.device,
            self.dtype,
        )

        self.teacher = teacher
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        # Determine weighting mode
        if distill_loss_weight is not None:
            # Mode 1: distill_weight specified - independent weights (task defaults to 1.0 if not set)
            self.distill_loss_weight = distill_loss_weight
            self.task_loss_weight = task_loss_weight if task_loss_weight is not None else 1.0
            if self.verbose:
                _logger.info(
                    f"FeatureDistillationTask: Independent weights - "
                    f"task_weight={self.task_loss_weight}, distill_weight={distill_loss_weight}"
                )
        elif task_loss_weight is not None:
            # Mode 2: only task_weight specified - complementary mode (distill = 1 - task)
            self.task_loss_weight = task_loss_weight
            self.distill_loss_weight = 1.0 - task_loss_weight
            if self.verbose:
                _logger.info(
                    f"FeatureDistillationTask: Complementary mode - "
                    f"task_weight={task_loss_weight}, distill_weight={self.distill_loss_weight}"
                )
        else:
            # Mode 3: neither specified - equal weights (both 1.0)
            self.distill_loss_weight = 1.0
            self.task_loss_weight = 1.0
            if self.verbose:
                _logger.info(
                    f"FeatureDistillationTask: Default equal weights - "
                    f"task_weight={self.task_loss_weight}, distill_weight={self.distill_loss_weight}"
                )

        # Auto-detect feature dimensions if not provided
        if student_feature_dim is None:
            student_feature_dim = self._detect_feature_dim(student_model)
        if teacher_feature_dim is None:
            teacher_feature_dim = self._detect_feature_dim(teacher.model)

        # Create projection layer if dimensions differ
        projection = None
        if student_feature_dim != teacher_feature_dim:
            if self.verbose:
                _logger.info(
                    f"Creating projection layer: {student_feature_dim} -> {teacher_feature_dim}"
                )
            projection = nn.Linear(student_feature_dim, teacher_feature_dim, device=self.device, dtype=self.dtype)
        else:
            if self.verbose:
                _logger.info("Feature dimensions match, no projection needed")

        self.trainable_module = FeatureDistillationTrainableModule(model=student_model, projection=projection)

        # Register student normalization values
        student_unwrapped = unwrap_model(student_model)
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

        if self.verbose:
            _logger.info(
                f"FeatureDistillationTask: "
                f"student_dim={student_feature_dim}, teacher_dim={teacher_feature_dim}"
            )

    @staticmethod
    def _detect_feature_dim(model: nn.Module) -> int:
        """Auto-detect feature dimension from model."""
        model = unwrap_model(model)

        if hasattr(model, 'head_hidden_size'):
            return model.head_hidden_size
        elif hasattr(model, 'num_features'):
            return model.num_features
        else:
            raise ValueError(
                "Cannot auto-detect feature dimension. Model must have "
                "'head_hidden_size' or 'num_features' attribute, or you must "
                "specify student_feature_dim and teacher_feature_dim explicitly."
            )

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs,
    ) -> 'FeatureDistillationTask':
        """Prepare task for distributed training.

        Wraps the trainable module (student + projection) in DistributedDataParallel
        (DDP) while leaving the frozen teacher model unwrapped.

        Args:
            device_ids: List of device IDs for DDP (e.g., [local_rank])
            **ddp_kwargs: Additional arguments passed to DistributedDataParallel

        Returns:
            self (for method chaining)
        """
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Use base class for DDP wrapping of trainable_module
        return super().prepare_distributed(device_ids=device_ids, **ddp_kwargs)

    def get_eval_task(self, use_ema: bool = True) -> ClassificationEvalTask:
        """Get evaluation task for classification.

        Args:
            use_ema: If True and EMA exists, use EMA weights for evaluation

        Returns:
            ClassificationEvalTask configured for the student model
        """
        return ClassificationEvalTask(self.get_trainable_module(use_ema))

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with feature distillation.

        Args:
            input: Input tensor [B, C, H, W]
            target: Target labels [B]

        Returns:
            Dictionary containing:
                - 'loss': Combined training loss (task + distillation)
                - 'output': Student logits (for metrics)
                - 'task_loss': Classification loss component
                - 'kd_loss': Feature distillation loss component
        """
        student_logits, student_features = self.trainable_module(input)
        task_loss = self.criterion(student_logits, target)

        with torch.no_grad():
            input_kd = self.teacher.normalize_input(input, self.student_mean, self.student_std)
            teacher_features = self.teacher(input_kd.detach(), return_features=True)

        kd_loss = F.mse_loss(student_features, teacher_features)
        total_loss = self.task_loss_weight * task_loss + self.distill_loss_weight * kd_loss

        return {
            'loss': total_loss,
            'output': student_logits,
            'task_loss': task_loss,
            'kd_loss': kd_loss,
        }
