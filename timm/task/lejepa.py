"""LeJEPA (Lean Joint-Embedding Predictive Architecture) training task.

LeJEPA is a self-supervised learning method that uses:
- SIGReg (Sketched Isotropic Gaussian Regularization) to constrain embeddings
- Invariance loss across multiple augmented views
- Single hyperparameter (lambda) for loss weighting

Reference:
    Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning
    Without the Heuristics", arXiv:2511.08544, 2025.
"""
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from timm.utils import unwrap_model
from .task import TrainingTask

_logger = logging.getLogger(__name__)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization loss.

    Statistical test that constrains embeddings to follow an isotropic Gaussian
    distribution using random slicing and the Epps-Pulley characteristic function test.

    The loss measures deviation from Gaussianity by comparing the empirical
    characteristic function to the theoretical Gaussian characteristic function
    along random 1D projections.

    Args:
        num_knots: Number of quadrature points for numerical integration (default: 17)
        num_slices: Number of random 1D projections for slicing (default: 256)
        t_max: Maximum integration bound (default: 3.0)

    Example:
        >>> sigreg = SIGReg(num_slices=256)
        >>> projections = torch.randn(4, 32, 128)  # [V, B, proj_dim]
        >>> loss = sigreg(projections)
    """

    def __init__(
            self,
            num_knots: int = 17,
            num_slices: int = 256,
            t_max: float = 3.0,
    ):
        super().__init__()
        self.num_slices = num_slices

        # Quadrature weights for trapezoidal integration on [0, t_max]
        # We use symmetry of ECF to integrate on [0, t_max] and double
        t = torch.linspace(0, t_max, num_knots, dtype=torch.float32)
        dt = t_max / (num_knots - 1)
        weights = torch.full((num_knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Trapezoidal rule endpoints

        # Gaussian characteristic function: exp(-t^2 / 2)
        phi_gaussian = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi_gaussian", phi_gaussian)
        self.register_buffer("weights", weights * phi_gaussian)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: Projected embeddings [V, B, proj_dim] or [B, proj_dim]
                  where V is number of views, B is batch size

        Returns:
            Scalar loss value
        """
        # Handle both [V, B, D] and [B, D] inputs
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)

        # Random projection directions (normalized)
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Project onto random directions: [V, B, num_slices]
        x_proj = proj @ A

        # Compute empirical characteristic function at quadrature points
        # x_t: [V, B, num_slices, num_knots]
        x_t = x_proj.unsqueeze(-1) * self.t

        # ECF components: E[cos(tx)] and E[sin(tx)]
        # Average over batch dimension (dim=-3 in [V, B, num_slices, num_knots])
        cos_mean = x_t.cos().mean(dim=-3)  # [V, num_slices, num_knots]
        sin_mean = x_t.sin().mean(dim=-3)  # [V, num_slices, num_knots]

        # Squared error from Gaussian ECF (which has sin component = 0)
        err = (cos_mean - self.phi_gaussian).square() + sin_mean.square()

        # Weighted integration and scale by batch size
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()


class LeJEPATrainableModule(nn.Module):
    """Trainable module for LeJEPA containing encoder and projector.

    Wraps the encoder model and adds a projector MLP. All trainable forward
    operations happen inside forward() for proper DDP/FSDP wrapping.

    Args:
        encoder: Backbone encoder model (any timm model)
        proj_dim: Output dimension of projector (default: 128)
        proj_hidden: Hidden dimension of projector MLP (default: 2048)
        proj_layers: Number of hidden layers in projector (default: 2)
    """

    def __init__(
            self,
            encoder: nn.Module,
            proj_dim: int = 128,
            proj_hidden: int = 2048,
            proj_layers: int = 2,
    ):
        super().__init__()
        self.encoder = encoder

        # Get encoder output dimension
        num_features = getattr(encoder, 'num_features', None)
        if num_features is None:
            raise ValueError(
                f"Encoder {encoder.__class__.__name__} must have 'num_features' attribute. "
                "Most timm models have this attribute."
            )

        # Build projector MLP: Linear -> BN -> GELU -> ... -> Linear
        layers = []
        in_dim = num_features
        for i in range(proj_layers):
            layers.extend([
                nn.Linear(in_dim, proj_hidden),
                nn.BatchNorm1d(proj_hidden),
                nn.GELU(),
            ])
            in_dim = proj_hidden
        layers.append(nn.Linear(proj_hidden, proj_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and projector.

        Args:
            x: Multi-view input images [B, V, C, H, W] where V is number of views

        Returns:
            Tuple of:
                - embeddings: Encoder outputs [B*V, num_features] (for optional probe)
                - projections: Projector outputs [V, B, proj_dim] (for loss computation)
        """
        if x.dim() != 5:
            raise ValueError(
                f"LeJEPA expects multi-view input [B, V, C, H, W] but got shape {x.shape}. "
                f"Make sure you're using create_multiview_train_loader() or a multi-view dataset."
            )

        B, V = x.shape[:2]

        # Flatten views into batch dimension
        x_flat = x.flatten(0, 1)  # [B*V, C, H, W]

        # Encode (use forward_features to get embeddings, not classifier logits)
        embeddings = self.encoder.forward_features(x_flat)  # [B*V, ...features]

        # Pool if needed (ViT returns [B, N, D], CNNs return [B, D] or [B, D, H, W])
        if embeddings.dim() == 3:
            # ViT-style: use CLS token or mean pool
            if hasattr(self.encoder, 'global_pool') and self.encoder.global_pool == 'avg':
                embeddings = embeddings.mean(dim=1)  # [B*V, D]
            else:
                embeddings = embeddings[:, 0]  # CLS token [B*V, D]
        elif embeddings.dim() == 4:
            # CNN-style: global average pool
            embeddings = embeddings.mean(dim=(2, 3))  # [B*V, D]

        # Project
        projections = self.projector(embeddings)  # [B*V, proj_dim]

        # Reshape projections for loss: [V, B, proj_dim]
        projections = projections.reshape(B, V, -1).transpose(0, 1)

        return embeddings, projections


class LeJEPATask(TrainingTask):
    """LeJEPA self-supervised training task.

    Combines SIGReg loss (ensures Gaussian embedding distribution) with
    invariance loss (views of same image should have similar embeddings).

    Loss = lambda * SIGReg + (1 - lambda) * Invariance

    Args:
        model: Encoder model (any timm model with num_features attribute)
        proj_dim: Projector output dimension (default: 128)
        proj_hidden: Projector hidden dimension (default: 2048)
        proj_layers: Number of projector hidden layers (default: 2)
        lamb: Loss weighting hyperparameter (default: 0.02)
            - Higher lambda = more weight on SIGReg (Gaussianity)
            - Lower lambda = more weight on invariance
        num_slices: Number of random projections for SIGReg (default: 256)
        num_knots: Quadrature points for SIGReg integration (default: 17)
        device: Device for task components
        dtype: Data type for task components
        verbose: Whether to log task configuration

    Example:
        >>> # With model name string
        >>> model = timm.create_model('vit_small_patch16_224', pretrained=False)
        >>> task = LeJEPATask(model, proj_dim=128, lamb=0.02)
        >>>
        >>> # Forward pass with multi-view input
        >>> x = torch.randn(32, 4, 3, 224, 224)  # [B, V, C, H, W]
        >>> output = task(x)
        >>> loss = output['loss']
    """

    def __init__(
            self,
            model: nn.Module,
            proj_dim: int = 128,
            proj_hidden: int = 2048,
            proj_layers: int = 2,
            lamb: float = 0.02,
            num_slices: int = 256,
            num_knots: int = 17,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)

        self.trainable_module = LeJEPATrainableModule(
            encoder=model,
            proj_dim=proj_dim,
            proj_hidden=proj_hidden,
            proj_layers=proj_layers,
        )
        self.sigreg = SIGReg(num_knots=num_knots, num_slices=num_slices)
        self.lamb = lamb

        # Move to device/dtype (encoder already on device, but projector and sigreg need moving)
        if device is not None or dtype is not None:
            self.trainable_module.projector.to(device=device, dtype=dtype)
            self.sigreg.to(device=device)

        if self.verbose:
            _logger.info(
                f"LeJEPATask: proj_dim={proj_dim}, proj_hidden={proj_hidden}, "
                f"proj_layers={proj_layers}, lambda={lamb}, num_slices={num_slices}"
            )

    @property
    def encoder(self) -> nn.Module:
        """Access the encoder model."""
        return unwrap_model(self.trainable_module).encoder

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        """Get state dict for checkpointing (encoder only, excludes projector)."""
        return self.encoder.state_dict()

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs,
    ) -> 'LeJEPATask':
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
        """Forward pass with LeJEPA loss computation.

        Args:
            input: Multi-view input images [B, V, C, H, W]
            target: Ignored (self-supervised task)

        Returns:
            Dictionary containing:
                - 'loss': Combined LeJEPA loss (for optimization)
                - 'output': Encoder embeddings [B*V, num_features] (for metrics/probing)
                - 'sigreg_loss': SIGReg component (for logging)
                - 'inv_loss': Invariance component (for logging)
        """
        embeddings, projections = self.trainable_module(input)

        # SIGReg loss - constrain to isotropic Gaussian
        sigreg_loss = self.sigreg(projections)

        # Invariance loss - views should have similar projections
        # proj_mean: [1, B, proj_dim], projections: [V, B, proj_dim]
        proj_mean = projections.mean(dim=0, keepdim=True)
        inv_loss = (proj_mean - projections).square().mean()

        # Combined loss with single hyperparameter
        total_loss = sigreg_loss * self.lamb + inv_loss * (1.0 - self.lamb)

        return {
            'loss': total_loss,
            'output': embeddings,
            'sigreg_loss': sigreg_loss,
            'inv_loss': inv_loss,
        }
