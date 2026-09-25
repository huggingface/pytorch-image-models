""" Distribution-Balanced Loss

Paper: `Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets`
    - Tong Wu, Qingqiu Huang, Ziwei Liu, Yu Wang, Dahua Lin, ECCV 2020 - https://arxiv.org/abs/2007.09654

Implemented from the paper formulation, numerically matching the DB configuration of the official
(unlicensed) reference at https://github.com/wutong16/DistributionBalancedLoss for binary targets.
"""
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionBalancedLoss(nn.Module):
    """Distribution-Balanced (DB) loss for long-tailed multi-label classification.

    Sigmoid BCE with two additions:
      * Re-balanced weighting: each (sample, class) term is weighted by the ratio of the class-level sampling
        probability 1 / n_i to the sample's instance-level one sum_j(y_j / n_j), which accounts for the label
        co-occurrence of each sample, smoothed as alpha + sigmoid(beta * (r - mu)).
      * Negative-tolerant regularization: logits are shifted by a class prior bias, and negative terms use a
        scaled logit, softplus(neg_scale * z) / neg_scale, so the many negatives of rare classes are not
        over-suppressed.
    An optional focal term (1 - p_t) ** gamma follows the reference DB configuration.

    Soft targets (Mixup/CutMix) weight the positive and negative terms, the re-balancing weights use the
    targets as given. Both match the reference for binary targets.

    Args:
        class_counts: Positive sample count per class over the training set.
        num_samples: Number of training samples the counts were taken over.
        map_alpha: Minimum weight of the re-balancing map.
        map_beta: Scale of the re-balancing map.
        map_mu: Threshold of the re-balancing map.
        neg_scale: Negative logit scale (lambda), 1 disables the negative-tolerant scaling.
        init_bias: Class prior logit shift factor (kappa), 0 disables it.
        focal_gamma: Focusing parameter, 0 disables the focal term (and focal_balance).
        focal_balance: Constant scale of the focal loss.
        smoothing: Label smoothing, moves each target towards 0.5.
        reduction: 'mean' averages over batch and classes (the reference scale), 'batchmean' sums over classes
            and averages over the batch, 'sum' sums over both, 'none' keeps the per-element loss.
    """

    def __init__(
            self,
            class_counts: Union[torch.Tensor, Sequence[float]],
            num_samples: int,
            map_alpha: float = 0.1,
            map_beta: float = 10.,
            map_mu: float = 0.2,
            neg_scale: float = 2.,
            init_bias: float = 0.05,
            focal_gamma: float = 2.,
            focal_balance: float = 2.,
            smoothing: float = 0.,
            reduction: str = 'mean',
    ):
        super().__init__()
        assert 0. <= smoothing < 1.
        assert neg_scale > 0
        if reduction not in ('mean', 'batchmean', 'sum', 'none'):
            raise ValueError(f"Unknown reduction '{reduction}', expected one of 'mean', 'batchmean', 'sum', 'none'.")
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or not counts.numel():
            raise ValueError('class_counts must be a non-empty 1-D sequence.')
        if num_samples < 2 or (counts < 0).any() or (counts > num_samples).any():
            raise ValueError('class_counts must be in [0, num_samples] with num_samples > 1.')
        # classes that are never (or always) positive would give infinite frequency / prior terms
        counts = counts.clamp(1., num_samples - 1.)
        self.map_alpha = map_alpha
        self.map_beta = map_beta
        self.map_mu = map_mu
        self.neg_scale = neg_scale
        self.focal_gamma = focal_gamma
        self.focal_balance = focal_balance
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer('freq_inv', 1. / counts)
        # prior logit log(p / (1 - p)) scaled by init_bias, divided by neg_scale as negatives multiply it back
        self.register_buffer('logit_bias', -torch.log(num_samples / counts - 1.) * init_bias / neg_scale)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input logits (batch_size, num_classes).
            target: Dense (multi-hot or soft) targets (batch_size, num_classes).
        """
        x = x.float()
        y = target.float()

        # Re-balanced weights, class-level over instance-level sampling probability of each (sample, class).
        # Samples without positives (instance probability 0) take the maximum weight, 1 + map_alpha.
        repeat_rate = (y * self.freq_inv).sum(dim=-1, keepdim=True)
        ratio = self.freq_inv / repeat_rate.clamp(min=torch.finfo(torch.float32).tiny)
        weight = torch.sigmoid(self.map_beta * (ratio - self.map_mu)) + self.map_alpha

        if self.smoothing:
            y = y * (1. - self.smoothing) + 0.5 * self.smoothing

        # Negative-tolerant regularization, prior shifted logits and scaled negative logits.
        z = x + self.logit_bias
        pos_loss = F.softplus(-z)
        neg_loss = F.softplus(z * self.neg_scale) / self.neg_scale
        if self.focal_gamma > 0:
            # 1 - p_t with p_t = sigmoid(z) for positives, sigmoid(-neg_scale * z) for negatives
            pos_loss = pos_loss * torch.sigmoid(-z) ** self.focal_gamma * self.focal_balance
            neg_loss = neg_loss * torch.sigmoid(z * self.neg_scale) ** self.focal_gamma * self.focal_balance
        loss = weight * (y * pos_loss + (1. - y) * neg_loss)

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'batchmean':
            return loss.sum() / x.shape[0]
        if self.reduction == 'sum':
            return loss.sum()
        return loss
