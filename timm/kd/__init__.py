"""Knowledge Distillation module for timm"""
from .distillation import DistillationTeacher, apply_kd_loss

__all__ = ['DistillationTeacher', 'apply_kd_loss']
