"""Training task abstractions for timm.

This module provides task-based abstractions for training loops where each task
encapsulates both the forward pass and loss computation, returning a dictionary
with loss components and outputs for logging.
"""
from .task import TrainingTask
from ._helpers import resume_task_checkpoint, load_task_ema_checkpoint
from .classification import (
    ClassificationTask,
    MultiLabelClassificationTask,
    create_classification_loss,
    resolve_classification_loss,
)
from .evaluator import ClassificationEvaluator, MultiLabelClassificationEvaluator, evaluation_sample_limit
from .distillation import DistillationTeacher, LogitDistillationTask, FeatureDistillationTask
from .token_distillation import TokenDistillationTeacher, TokenDistillationTask

__all__ = [
    'TrainingTask',
    'resume_task_checkpoint',
    'load_task_ema_checkpoint',
    'ClassificationTask',
    'MultiLabelClassificationTask',
    'create_classification_loss',
    'resolve_classification_loss',
    'ClassificationEvaluator',
    'MultiLabelClassificationEvaluator',
    'evaluation_sample_limit',
    'DistillationTeacher',
    'LogitDistillationTask',
    'FeatureDistillationTask',
    'TokenDistillationTeacher',
    'TokenDistillationTask',
]
