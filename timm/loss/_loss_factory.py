""" Classification loss factory

Hacked together by / Copyright 2026 Ross Wightman
"""
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .binary_cross_entropy import BinaryCrossEntropy
from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .distribution_balanced_loss import DistributionBalancedLoss
from .jsd import JsdCrossEntropy
from .poly_loss import PolyBinaryCrossEntropy, PolyCrossEntropy
from .two_way_loss import TwoWayLoss
from .zlpr_loss import ZlprLoss

LOSS_TYPES = ('ce', 'bce', 'asl', 'poly', 'jsd', 'twoway', 'zlpr', 'db')
MULTI_LABEL_LOSS_TYPES = ('bce', 'asl', 'poly', 'twoway', 'zlpr', 'db')


def _as_weight(
        weight: Optional[Union[torch.Tensor, float, Sequence[float]]],
) -> Optional[Union[torch.Tensor, float]]:
    if weight is None or isinstance(weight, (float, int)):
        return weight
    return torch.as_tensor(weight, dtype=torch.float32)


def create_classification_loss(
        loss_type: Optional[str] = None,
        multi_label: bool = False,
        smoothing: float = 0.,
        soft_targets: bool = False,
        class_weight: Optional[Union[torch.Tensor, float, Sequence[float]]] = None,
        pos_weight: Optional[Union[torch.Tensor, float, Sequence[float]]] = None,
        bce_target_thresh: Optional[float] = None,
        bce_sum: bool = False,
        asl_gamma_pos: float = 1.,
        asl_gamma_neg: float = 4.,
        asl_clip: Optional[float] = 0.05,
        asl_reduction: str = 'batchmean',
        poly_epsilon: float = 2.,
        poly_gamma: float = 0.,
        poly_alpha: Optional[float] = None,
        jsd_splits: int = 0,
        twoway_tp: float = 4.,
        twoway_tn: float = 1.,
        class_counts: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        num_samples: Optional[int] = None,
        db_neg_scale: float = 2.,
        db_init_bias: float = 0.05,
        db_focal_gamma: float = 2.,
) -> nn.Module:
    """Create the training criterion for a classification task.

    Args:
        loss_type: One of LOSS_TYPES, None selects 'bce' for multi-label and 'ce' for single-label targets.
            'ce': softmax cross-entropy (label smoothing / soft target variants as needed),
            'bce': binary cross-entropy (bce_* args),
            'asl': Asymmetric Loss, AsymmetricLossMultiLabel for multi-label targets, else the softmax
                AsymmetricLossSingleLabel, class index targets only, no Mixup/CutMix (asl_* args),
            'poly': Poly-1 loss, PolyBinaryCrossEntropy (sigmoid) for multi-label targets, else the softmax
                PolyCrossEntropy (poly_* args),
            'jsd': Jensen-Shannon divergence + CE over augmentation splits, single-label (jsd_splits),
            'twoway': Two-way loss, binary multi-label targets only (twoway_* args),
            'zlpr': ZLPR loss, binary multi-label targets only,
            'db': Distribution-Balanced loss, multi-label (class_counts, num_samples, db_* args).
        multi_label: Targets are dense independent binary labels shaped (B, C).
        smoothing: Label smoothing factor. Ignored when soft_targets is set, since Mixup/CutMix
            already apply it while producing the soft targets.
        soft_targets: Targets arrive as dense soft distributions from Mixup/CutMix rather than class indices.
        class_weight: Per-class loss weight, a single value or one per class ('bce' and multi-label 'asl').
        pos_weight: Positive term weight, a single value or one per class ('bce' and multi-label 'asl').
        bce_target_thresh: Binarize soft targets above this value ('bce' only).
        bce_sum: Sum the loss over classes before averaging over the batch ('bce' only).
        asl_gamma_pos: Focusing parameter for positive targets ('asl' only).
        asl_gamma_neg: Focusing parameter for negative targets ('asl' only).
        asl_clip: Probability margin shifted onto negatives, None or 0 disables (multi-label 'asl' only).
        asl_reduction: 'batchmean' (sum over classes, mean over batch), 'sum' (original ASL), or 'mean'
            (multi-label 'asl' only, single-label ASL always uses 'batchmean').
        poly_epsilon: Poly-1 coefficient, 0 reduces to CE / BCE ('poly' only).
        poly_gamma: Focusing parameter, 0 disables the focal term (multi-label 'poly' only).
        poly_alpha: Positive / negative balance weight, None disables it (multi-label 'poly' only).
        jsd_splits: Number of augmentation splits, must be > 1 ('jsd' only).
        twoway_tp: Positive logit temperature ('twoway' only).
        twoway_tn: Negative logit temperature ('twoway' only).
        class_counts: Positive training sample count per class ('db' only).
        num_samples: Number of training samples class_counts were taken over ('db' only).
        db_neg_scale: Negative-tolerant logit scale, 1 disables ('db' only).
        db_init_bias: Class prior logit shift factor, 0 disables ('db' only).
        db_focal_gamma: Focusing parameter, 0 disables the focal term ('db' only).
    """
    loss_type = loss_type or ('bce' if multi_label else 'ce')
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Unknown loss type '{loss_type}', expected one of {LOSS_TYPES}.")
    if multi_label and loss_type not in MULTI_LABEL_LOSS_TYPES:
        raise ValueError(
            f"Loss type '{loss_type}' does not support multi-label targets, expected one of {MULTI_LABEL_LOSS_TYPES}.")
    if not multi_label and loss_type in ('twoway', 'zlpr', 'db'):
        raise ValueError(f"Loss type '{loss_type}' requires multi-label targets.")
    class_weight = _as_weight(class_weight)
    pos_weight = _as_weight(pos_weight)
    if (class_weight is not None or pos_weight is not None) and not (
            loss_type == 'bce' or (loss_type == 'asl' and multi_label)):
        raise ValueError('Class and positive weights are only supported by bce and multi-label asl losses.')
    if loss_type != 'bce' and (bce_target_thresh is not None or bce_sum):
        raise ValueError('BCE target threshold and sum options only apply to the bce loss.')
    if loss_type != 'db' and (class_counts is not None or num_samples is not None):
        raise ValueError('Class counts are only used by the db loss.')

    if loss_type == 'ce':
        if soft_targets:
            return SoftTargetCrossEntropy()
        if smoothing:
            return LabelSmoothingCrossEntropy(smoothing=smoothing)
        return nn.CrossEntropyLoss()
    if loss_type == 'bce':
        return BinaryCrossEntropy(
            smoothing=0. if soft_targets else smoothing,
            smooth_dense=multi_label,
            target_threshold=bce_target_thresh,
            weight=torch.as_tensor(class_weight, dtype=torch.float32) if class_weight is not None else None,
            sum_classes=bce_sum,
            pos_weight=pos_weight,
        )
    if loss_type == 'asl':
        if multi_label:
            return AsymmetricLossMultiLabel(
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                clip=asl_clip,
                smoothing=0. if soft_targets else smoothing,
                reduction=asl_reduction,
                weight=torch.as_tensor(class_weight, dtype=torch.float32) if class_weight is not None else None,
                pos_weight=pos_weight,
            )
        if soft_targets:
            raise ValueError('Single-label ASL requires class index targets, it does not support Mixup/CutMix.')
        if asl_reduction != 'batchmean':
            raise ValueError("Single-label ASL only supports asl_reduction='batchmean'.")
        return AsymmetricLossSingleLabel(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, eps=smoothing)
    if loss_type == 'poly':
        smoothing = 0. if soft_targets else smoothing
        if multi_label:
            return PolyBinaryCrossEntropy(
                epsilon=poly_epsilon, gamma=poly_gamma, alpha=poly_alpha, smoothing=smoothing)
        if poly_gamma or poly_alpha is not None:
            raise ValueError('Poly gamma and alpha only apply to the multi-label (sigmoid) poly loss.')
        return PolyCrossEntropy(epsilon=poly_epsilon, smoothing=smoothing)
    if loss_type == 'jsd':
        if jsd_splits < 2:
            raise ValueError('JSD loss requires jsd_splits > 1 (augmentation splits).')
        return JsdCrossEntropy(num_splits=jsd_splits, smoothing=smoothing)
    if loss_type in ('twoway', 'zlpr'):
        if soft_targets or smoothing:
            raise ValueError(
                f"Loss type '{loss_type}' requires binary targets, it does not support Mixup/CutMix or smoothing.")
        return TwoWayLoss(tp=twoway_tp, tn=twoway_tn) if loss_type == 'twoway' else ZlprLoss()
    # db
    if class_counts is None or num_samples is None:
        raise ValueError('The db loss requires class_counts and num_samples (see class_weights.py).')
    return DistributionBalancedLoss(
        class_counts,
        num_samples,
        neg_scale=db_neg_scale,
        init_bias=db_init_bias,
        focal_gamma=db_focal_gamma,
        smoothing=0. if soft_targets else smoothing,
    )
