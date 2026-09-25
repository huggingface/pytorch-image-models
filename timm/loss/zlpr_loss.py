""" ZLPR (zero-bounded log-sum-exp & pairwise rank-based) Loss

Paper: `ZLPR: A Novel Loss for Multi-label Classification` - https://arxiv.org/abs/2208.02955
"""
import torch
import torch.nn as nn


class ZlprLoss(nn.Module):
    """ZLPR loss, the multi-label extension of softmax cross-entropy.

    Per sample, log(1 + sum_neg exp(s_j)) + log(1 + sum_pos exp(-s_i)). Every positive logit is pushed above
    every negative one and 0 acts as the decision threshold, so logits > 0 (sigmoid > 0.5) predict positives.
    No hyperparameters, and the loss does not grow with the number of negative classes like per-class BCE.

    Targets are binarized, entries above 0.5 are positives.
    """

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input logits (batch_size, num_classes).
            target: Dense binary targets (batch_size, num_classes).
        """
        x = x.float()
        pos = target > 0.5
        # The appended zero logit keeps each log-sum-exp finite, so masking with -inf is safe here.
        zeros = x.new_zeros(x.shape[:-1] + (1,))
        neg_term = torch.logsumexp(torch.cat([x.masked_fill(pos, float('-inf')), zeros], dim=-1), dim=-1)
        pos_term = torch.logsumexp(torch.cat([(-x).masked_fill(~pos, float('-inf')), zeros], dim=-1), dim=-1)
        return (neg_term + pos_term).mean()
