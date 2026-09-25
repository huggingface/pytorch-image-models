""" Two-way Multi-Label Loss

Paper: `Two-way Multi-Label Loss` - Takumi Kobayashi, CVPR 2023
https://staff.aist.go.jp/takumi.kobayashi/publication/2023/CVPR2023.pdf

Implemented from the paper formulation, numerically matching the official (unlicensed) reference at
https://github.com/tk1980/TwoWayMultiLabelLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Finite fill for masked logits, a row / column without any valid entry must not produce inf - inf in backward.
_MASK_VALUE = -1e4


class TwoWayLoss(nn.Module):
    """Two-way multi-label loss.

    Contrasts the hardest (soft-max) negative logit with the hardest (soft-min) positive logit in two directions,
    across the classes of each sample and across the samples of each class in the batch. Only samples / classes
    with at least one positive contribute to the respective direction.

    Targets are binarized, entries above 0.5 are positives. Logits are ranking scores, they are not calibrated
    as independent sigmoid probabilities.

    Args:
        tp: Temperature of the soft-min over positive logits.
        tn: Temperature of the soft-max over negative logits.
    """

    def __init__(self, tp: float = 4., tn: float = 1.):
        super().__init__()
        assert tp > 0 and tn > 0
        self.tp = tp
        self.tn = tn

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input logits (batch_size, num_classes).
            target: Dense binary targets (batch_size, num_classes).
        """
        x = x.float()
        pos = target > 0.5
        pos_logits = (-x / self.tp).masked_fill(~pos, _MASK_VALUE)
        neg_logits = (x / self.tn).masked_fill(pos, _MASK_VALUE)

        loss = x.new_zeros(())
        for dim in (0, 1):  # 0: over the samples of each class, 1: over the classes of each sample
            valid = pos.any(dim=dim)
            hard_pos = torch.logsumexp(pos_logits, dim=dim) * self.tp
            hard_neg = torch.logsumexp(neg_logits, dim=dim) * self.tn
            # mean over valid entries, zero (but still attached to the graph) if there are none
            loss = loss + F.softplus(hard_neg + hard_pos)[valid].sum() / valid.sum().clamp(min=1)
        return loss
