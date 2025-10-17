"""Knowledge Distillation helpers for training with a teacher model."""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T

from timm.models import create_model


_logger = logging.getLogger(__name__)


class DistillationTeacher(nn.Module):
    """Wrapper for a teacher model used in knowledge distillation.

    Creates and manages a pre-trained teacher model for knowledge distillation,
    handling model compilation and normalization differences between teacher and student.

    Args:
        model_name: Name of the teacher model to create
        num_classes: Number of output classes
        in_chans: Number of input channels
        device: Device to place the model on (default: 'cuda')
        dtype: Model dtype (default: None, uses float32)
    """

    def __init__(
            self,
            model_name: str,
            num_classes: int,
            in_chans: int = 3,
            pretrained_path: Optional[str] = None,
            device: torch.device = torch.device('cuda'),
            dtype: torch.dtype = None,
    ):
        super().__init__()

        _logger.info(f"Creating KD teacher model: '{model_name}'")

        pretrained_kwargs = {'pretrained': True}
        if pretrained_path:
            # specify a local checkpoint path to load pretrained weights from
            pretrained_kwargs['pretrained_cfg_overlay'] = dict(
                file=pretrained_path,
                num_classes=num_classes,  # needed to avoid head adaptation?
            )

        model_kd = create_model(
            model_name=model_name,
            num_classes=num_classes,
            in_chans=in_chans,
            **pretrained_kwargs,
        )

        model_kd = model_kd.to(device=device, dtype=dtype)
        model_kd.eval()

        try:
            model_kd = torch.compile(model_kd)
            _logger.info("torch.compile applied successfully to KD teacher model")
        except Exception as e:
            _logger.warning(f"torch.compile failed with error {e}, continuing without compilation")

        self.model = model_kd
        self.mean_model_kd = model_kd.pretrained_cfg['mean']
        self.std_model_kd = model_kd.pretrained_cfg['std']

    def normalize_input(
        self,
        input: torch.Tensor,
        student_model: nn.Module,
    ) -> torch.Tensor:
        """Normalize input to match teacher's expected normalization.

        Handles different normalization between teacher and student models by
        converting the student's normalized input to the teacher's expected format.

        Args:
            input: Input tensor (already normalized for student)
            student_model: Student model to extract normalization params from

        Returns:
            Input tensor normalized for the teacher model
        """
        if hasattr(student_model, 'module'):
            model_s = student_model.module
        else:
            model_s = student_model

        mean_student = model_s.pretrained_cfg['mean']
        std_student = model_s.pretrained_cfg['std']

        input_kd = input
        if mean_student != self.mean_model_kd or std_student != self.std_model_kd:
            # Compute normalized std and mean transformations
            std = tuple(t_std / s_std for t_std, s_std in zip(self.std_model_kd, std_student))
            transform_std = T.Normalize(mean=(0, 0, 0), std=std)

            mean = tuple(t_mean - s_mean for t_mean, s_mean in zip(self.mean_model_kd, mean_student))
            transform_mean = T.Normalize(mean=mean, std=(1, 1, 1))

            input_kd = transform_mean(transform_std(input))

        return input_kd


def apply_kd_loss(
        loss: torch.Tensor,
        student_output: torch.Tensor,
        input: torch.Tensor,
        student_model: nn.Module,
        teacher_model: DistillationTeacher,
        alpha_kd: float,
        use_kd_only: bool = False,
) -> torch.Tensor:
    """Apply knowledge distillation loss.

    Computes KL divergence between student and teacher outputs and combines
    with the base loss (or replaces it if use_kd_only is True).

    Args:
        loss: Base loss (e.g., cross-entropy with labels)
        student_output: Logits from student model
        input: Input tensor (already normalized for student)
        student_model: Student model being trained
        teacher_model: Teacher model for distillation
        alpha_kd: Weight for the KD loss component
        use_kd_only: If True, only use KD loss (ignore base loss)

    Returns:
        Combined loss with KD component
    """
    # Student probability calculation
    prob_s = torch.nn.functional.log_softmax(student_output, dim=-1)

    # Teacher probability calculation
    with torch.no_grad():
        input_kd = teacher_model.normalize_input(input, student_model)
        out_t = teacher_model.model(input_kd.detach())
        prob_t = torch.nn.functional.softmax(out_t, dim=-1)

    # Compute KL divergence loss
    kd_loss = alpha_kd * torch.nn.functional.kl_div(prob_s, prob_t, reduction='batchmean')

    if use_kd_only:
        return kd_loss
    else:
        return loss + kd_loss
