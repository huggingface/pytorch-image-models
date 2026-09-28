from typing import Optional, Union

import torch
import torch.nn as nn


class AsymmetricLossMultiLabel(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification, https://arxiv.org/abs/2009.14119

    Args:
        gamma_neg: Focusing parameter for negative targets.
        gamma_pos: Focusing parameter for positive targets.
        clip: Probability margin shifted onto negatives (asymmetric clipping), disabled if None or 0.
        eps: Minimum probability clamped to before the log.
        disable_torch_grad_focal_loss: Don't backprop through the focusing weights.
        smoothing: Label smoothing, moves each (dense, independent binary) target towards 0.5.
        reduction: 'batchmean' sums over classes and averages over the batch, 'sum' sums over batch and classes
            (the original ASL reduction, scales with batch size), 'mean' averages over batch and classes,
            'none' keeps the per-element loss.
        weight: Per-class rescaling weight.
        pos_weight: Weight of the positive term, a single value or one per class.
    """
    def __init__(
            self,
            gamma_neg: float = 4,
            gamma_pos: float = 1,
            clip: Optional[float] = 0.05,
            eps: float = 1e-8,
            disable_torch_grad_focal_loss: bool = False,
            smoothing: float = 0.,
            reduction: str = 'batchmean',
            weight: Optional[torch.Tensor] = None,
            pos_weight: Optional[Union[torch.Tensor, float]] = None,
    ):
        super(AsymmetricLossMultiLabel, self).__init__()
        assert 0. <= smoothing < 1.
        if reduction not in ('batchmean', 'sum', 'mean', 'none'):
            raise ValueError(f"Unknown reduction '{reduction}', expected one of 'batchmean', 'sum', 'mean', 'none'.")

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.smoothing = smoothing
        self.reduction = reduction
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input logits (batch_size, num_classes).
            y: Dense (multi-hot or soft) targets (batch_size, num_classes).
        """
        # Compute in float32, low precision (fp16) probabilities underflow the eps clamp and produce NaN.
        x = x.float()
        y = y.float()
        if self.smoothing:
            y = y * (1. - self.smoothing) + 0.5 * self.smoothing

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        if self.pos_weight is not None:
            los_pos = los_pos * self.pos_weight
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.disable_torch_grad_focal_loss):
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        loss = -loss
        if self.weight is not None:
            loss = loss * self.weight
        if self.reduction == 'batchmean':
            return loss.sum() / x.shape[0]
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss


class AsymmetricLossSingleLabel(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(AsymmetricLossSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (1-hot vector)
        """

        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
