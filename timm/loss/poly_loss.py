""" PolyLoss (Poly-1)

Paper: `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions`
    - Zhaoqi Leng, Mingxing Tan, Chenxi Liu, Ekin Dogus Cubuk, Xiaojie Shi, Shuyang Cheng, Dragomir Anguelov
    - ICLR 2022, https://arxiv.org/abs/2204.12511

Poly-1 adds epsilon * (1 - p_t) to the loss, adjusting the weight of the leading polynomial term of the
cross-entropy (or focal loss) expansion. Positive epsilon increases prediction confidence, the paper used
epsilon = 2 for EfficientNetV2 ImageNet-21K pretraining and ImageNet-1K finetuning, and epsilon = -1 with
focal loss for detection.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyCrossEntropy(nn.Module):
    """Poly-1 softmax cross-entropy, CE + epsilon * (1 - p_t), for single-label classification.

    p_t is the target-weighted softmax probability sum_c(t_c * p_c), so label smoothing and soft (Mixup/CutMix)
    targets apply to both terms, as in the reference.

    Args:
        epsilon: Poly-1 coefficient, 0 is plain cross-entropy.
        smoothing: Label smoothing applied to class index targets. Dense targets are assumed to be softened
            upstream (e.g. by Mixup/CutMix).
        reduction: 'mean' or 'sum' over the batch, or 'none' for per-sample losses.
    """

    def __init__(self, epsilon: float = 2., smoothing: float = 0., reduction: str = 'mean'):
        super().__init__()
        assert 0. <= smoothing < 1.
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"Unknown reduction '{reduction}', expected one of 'mean', 'sum', 'none'.")
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input logits (batch_size, num_classes).
            target: Class indices (batch_size,) or dense target distributions (batch_size, num_classes).
        """
        log_probs = F.log_softmax(x.float(), dim=-1)
        if target.shape != x.shape:
            num_classes = x.shape[-1]
            target = F.one_hot(target.long(), num_classes).float()
            if self.smoothing:
                target = target * (1. - self.smoothing) + self.smoothing / num_classes
        target = target.float()
        loss = -(target * log_probs).sum(dim=-1)
        pt = (target * log_probs.exp()).sum(dim=-1)
        loss = loss + self.epsilon * (1. - pt)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class PolyBinaryCrossEntropy(nn.Module):
    """Poly-1 sigmoid (focal) loss, FL + epsilon * (1 - p_t) ** (gamma + 1), for multi-label classification.

    With gamma = 0 this is BCE + epsilon * (1 - p_t) per class. With alpha set, positives are weighted by alpha
    and negatives by 1 - alpha in both terms, as in the reference.

    Args:
        epsilon: Poly-1 coefficient, 0 is plain BCE (or focal loss).
        gamma: Focusing parameter, 0 disables the focal term.
        alpha: Positive / negative balance weight, None disables it.
        smoothing: Label smoothing, moves each (dense, independent binary) target towards 0.5.
        reduction: 'mean' averages over batch and classes (as BCE), 'batchmean' sums over classes and averages
            over the batch, 'sum' sums over both, 'none' keeps the per-element loss.
    """

    def __init__(
            self,
            epsilon: float = 2.,
            gamma: float = 0.,
            alpha: Optional[float] = None,
            smoothing: float = 0.,
            reduction: str = 'mean',
    ):
        super().__init__()
        assert 0. <= smoothing < 1.
        assert alpha is None or 0. <= alpha <= 1.
        if reduction not in ('mean', 'batchmean', 'sum', 'none'):
            raise ValueError(f"Unknown reduction '{reduction}', expected one of 'mean', 'batchmean', 'sum', 'none'.")
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input logits (batch_size, num_classes).
            target: Dense (multi-hot or soft) targets (batch_size, num_classes).
        """
        x = x.float()
        y = target.float()
        if self.smoothing:
            y = y * (1. - self.smoothing) + 0.5 * self.smoothing
        p = torch.sigmoid(x)
        one_minus_pt = 1. - (y * p + (1. - y) * (1. - p))
        loss = F.binary_cross_entropy_with_logits(x, y, reduction='none')
        poly = one_minus_pt
        if self.gamma:
            focal = one_minus_pt ** self.gamma
            loss = loss * focal
            poly = poly * focal
        loss = loss + self.epsilon * poly
        if self.alpha is not None:
            loss = loss * (y * self.alpha + (1. - y) * (1. - self.alpha))

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'batchmean':
            return loss.sum() / x.shape[0]
        if self.reduction == 'sum':
            return loss.sum()
        return loss
