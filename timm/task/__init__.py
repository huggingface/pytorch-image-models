"""Training task abstractions for timm.

This module provides task-based abstractions for training loops where each task
encapsulates both the forward pass and loss computation, returning a dictionary
with loss components and outputs for logging.
"""
from .task import TrainingTask
from .classification import ClassificationTask
from .distillation import DistillationTeacher, LogitDistillationTask, FeatureDistillationTask
from .token_distillation import TokenDistillationTeacher, TokenDistillationTask
from .lejepa import SIGReg, LeJEPATrainableModule, LeJEPATask
from .nepa import ResidualMlpBlock, PixelDecoder, NEPATrainableModule, NEPATask

__all__ = [
    # Base
    'TrainingTask',
    # Classification
    'ClassificationTask',
    # Distillation
    'DistillationTeacher',
    'LogitDistillationTask',
    'FeatureDistillationTask',
    'TokenDistillationTeacher',
    'TokenDistillationTask',
    # Self-supervised
    'SIGReg',
    'LeJEPATrainableModule',
    'LeJEPATask',
    'ResidualMlpBlock',
    'PixelDecoder',
    'NEPATrainableModule',
    'NEPATask',
]
