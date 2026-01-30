"""Training task abstractions for timm.

This module provides task-based abstractions for training loops where each task
encapsulates both the forward pass and loss computation, returning a dictionary
with loss components and outputs for logging.
"""
from .task import TrainingTask
from .classification import ClassificationTask
from .distillation import DistillationTeacher, LogitDistillationTask, FeatureDistillationTask
from .token_distillation import TokenDistillationTeacher, TokenDistillationTask
from .recursive_supervision import RecursiveSupervisionTask

__all__ = [
    'TrainingTask',
    'ClassificationTask',
    'DistillationTeacher',
    'LogitDistillationTask',
    'FeatureDistillationTask',
    'TokenDistillationTeacher',
    'TokenDistillationTask',
    'RecursiveSupervisionTask',
]
