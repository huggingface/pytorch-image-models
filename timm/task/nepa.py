"""NEPA (Next Embedding Prediction Architecture) training task.

NEPA is a self-supervised learning method for Vision Transformers that trains
the model to predict input embeddings from output embeddings, either at shifted
positions (predict next token) or same positions.

This implementation provides a wrapper that works with existing timm ViTs
(vision_transformer.py, eva.py) without requiring architecture modifications.
"""
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from timm.utils import unwrap_model
from .task import TrainingTask

_logger = logging.getLogger(__name__)


def prediction_loss(
        h_in: torch.Tensor,
        h_out: torch.Tensor,
        shift: bool = True,
) -> torch.Tensor:
    """NEPA prediction loss - negative cosine similarity between embeddings.

    Computes the similarity between input embeddings (target) and output
    embeddings (prediction), optionally with a position shift for next-token
    prediction.

    Args:
        h_in: Input embeddings [B, N, D] (target, will be detached)
        h_out: Output embeddings [B, N, D] (prediction)
        shift: If True, compare h_out[:, :-1] with h_in[:, 1:] (predict next).
               If False, compare at same positions.

    Returns:
        Scalar loss (negative cosine similarity, lower is better)
    """
    # Detach target to prevent gradient flow
    h_in = h_in.detach()

    if shift:
        # Predict next position: output[t] predicts input[t+1]
        p = h_out[:, :-1, :]  # Predictions
        z = h_in[:, 1:, :]    # Targets (shifted)
    else:
        # Same position prediction
        p = h_out
        z = h_in

    # L2 normalize
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    # Negative cosine similarity (mean over all positions and batches)
    loss = -(p * z).sum(dim=-1).mean()

    return loss


class NEPATrainableModule(nn.Module):
    """Trainable module wrapper for NEPA training with timm ViTs.

    Wraps a timm Vision Transformer to provide access to both input embeddings
    (after patch embedding + position embedding) and output embeddings (after
    transformer blocks) needed for NEPA's prediction loss.

    Uses the model's forward_intermediates() method with return_input_embeddings
    for clean integration without replicating forward logic.

    Compatible with:
        - vision_transformer.py (VisionTransformer, VisionTransformerDistilled, etc.)
        - eva.py (Eva, etc.)
        - Other ViTs with forward_intermediates() supporting return_input_embeddings

    Args:
        model: A timm ViT model with forward_intermediates method

    Attributes:
        model: The wrapped ViT model
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        if not hasattr(model, 'forward_intermediates'):
            raise ValueError(
                f"Model {model.__class__.__name__} must have 'forward_intermediates' method. "
                f"NEPATrainableModule is compatible with timm ViTs (VisionTransformer, Eva, etc.)"
            )
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning output and input embeddings.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Tuple of:
                - output_embeddings: After transformer blocks + norm [B, N, D]
                - input_embeddings: After pos embed, before blocks [B, N, D]
        """
        result = self.model.forward_intermediates(
            x,
            return_input_embeddings=True,
            output_dict=True,
        )

        output_embeddings = result['image_features']
        input_embeddings = result['input_embeddings']

        return output_embeddings, input_embeddings


class NEPATask(TrainingTask):
    """NEPA (Next Embedding Prediction Architecture) training task.

    Self-supervised task that trains a ViT to predict input embeddings from
    output embeddings. The prediction target is the input embedding at either
    the next position (shift=True) or same position (shift=False).

    This task wraps existing timm ViTs without requiring architecture changes.

    Args:
        model: A timm ViT model (VisionTransformer, Eva, etc.)
        shift: If True, predict next position (h_out[t] -> h_in[t+1]).
               If False, predict same position (default: True)
        device: Device for task components
        dtype: Data type for task components
        verbose: Whether to log task configuration

    Example:
        >>> # With a timm ViT
        >>> model = timm.create_model('vit_base_patch16_224', pretrained=False)
        >>> task = NEPATask(model, shift=True)
        >>>
        >>> # Forward pass
        >>> x = torch.randn(32, 3, 224, 224)
        >>> output = task(x)
        >>> loss = output['loss']
        >>>
        >>> # With Eva model
        >>> model = timm.create_model('eva02_base_patch14_224', pretrained=False)
        >>> task = NEPATask(model, shift=True)
    """

    def __init__(
            self,
            model: nn.Module,
            shift: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)

        self.trainable_module = NEPATrainableModule(model)
        self.shift = shift

        if self.verbose:
            _logger.info(f"NEPATask: shift={shift}, model={model.__class__.__name__}")

    @property
    def model(self) -> nn.Module:
        """Access the wrapped ViT model."""
        return unwrap_model(self.trainable_module).model

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        """Get state dict for checkpointing (model weights only)."""
        return unwrap_model(self.trainable_module).model.state_dict()

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs,
    ) -> 'NEPATask':
        """Prepare task for distributed training.

        Wraps the trainable module in DistributedDataParallel (DDP).

        Args:
            device_ids: List of device IDs for DDP (e.g., [local_rank])
            **ddp_kwargs: Additional arguments passed to DistributedDataParallel

        Returns:
            self (for method chaining)
        """
        self.trainable_module = DDP(
            self.trainable_module,
            device_ids=device_ids,
            **ddp_kwargs
        )
        return self

    def forward(
            self,
            input: torch.Tensor,
            target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with NEPA prediction loss.

        Args:
            input: Input images [B, C, H, W]
            target: Ignored (self-supervised task)

        Returns:
            Dictionary containing:
                - 'loss': NEPA prediction loss (for optimization)
                - 'output': Output embeddings [B, N, D] (for downstream use)
                - 'input_embeddings': Input embeddings [B, N, D] (for analysis)
        """
        output_embeddings, input_embeddings = self.trainable_module(input)

        loss = prediction_loss(input_embeddings, output_embeddings, shift=self.shift)

        return {
            'loss': loss,
            'output': output_embeddings,
            'input_embeddings': input_embeddings,
        }
