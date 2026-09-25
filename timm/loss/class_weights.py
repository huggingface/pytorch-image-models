""" Per-class loss weights derived from label frequencies

Class stats (per-class positive counts and the sample count, written by `class_weights.py` in the repo root)
are loaded with `load_class_stats()`. `compute_class_weights()` turns them into BCE / ASL `pos_weight` or
`class_weight` vectors, `load_class_weights()` parses explicit weights (a number, a comma separated list, or a
file), and `resolve_class_weights()` accepts either form, as given to train.py.
"""
import json
import os
from typing import Optional, Sequence, Tuple, Union

import torch

# method -> the loss argument its weights are for
CLASS_WEIGHT_METHOD_KINDS = {'neg_pos': 'pos_weight', 'inv_freq': 'class_weight', 'effective_num': 'class_weight'}
CLASS_WEIGHT_METHODS = tuple(CLASS_WEIGHT_METHOD_KINDS)


def compute_class_weights(
        pos_counts: Union[torch.Tensor, Sequence[float]],
        num_samples: int,
        method: str = 'neg_pos',
        power: float = 1.,
        beta: float = 0.999,
        max_weight: Optional[float] = None,
) -> Tuple[str, torch.Tensor]:
    """Compute per-class loss weights from positive label counts.

    Classes without any positives are treated as having one, and every weight is capped at max_weight if set.

    Args:
        pos_counts: Number of positive samples per class (the class frequency for single-label targets).
        num_samples: Number of samples the counts were taken over.
        method: One of
            'neg_pos': pos_weight = (negatives / positives) ** power, balances positive and negative terms
                of each class,
            'inv_freq': class_weight proportional to (num_samples / positives) ** power, mean 1,
            'effective_num': class-balanced class_weight from the effective number of samples
                (1 - beta ** n) / (1 - beta), https://arxiv.org/abs/1901.05555, mean 1.
        power: Exponent for 'neg_pos' and 'inv_freq', 0.5 gives a softer (square root) re-weighting.
        beta: Effective number hyperparameter for 'effective_num'.
        max_weight: Cap applied to the final weights.

    Returns:
        The loss argument the weights are for ('pos_weight' or 'class_weight') and the float32 weights.
    """
    counts = torch.as_tensor(pos_counts, dtype=torch.float64)
    if counts.ndim != 1 or not counts.numel():
        raise ValueError('pos_counts must be a non-empty 1-D sequence.')
    if num_samples <= 0 or (counts < 0).any() or (counts > num_samples).any():
        raise ValueError('pos_counts must be in [0, num_samples] with a positive num_samples.')
    counts = counts.clamp(min=1.)
    if method not in CLASS_WEIGHT_METHOD_KINDS:
        raise ValueError(f"Unknown class weight method '{method}', expected one of {CLASS_WEIGHT_METHODS}.")
    kind = CLASS_WEIGHT_METHOD_KINDS[method]
    if method == 'neg_pos':
        weights = ((num_samples - counts).clamp(min=0.) / counts) ** power
    elif method == 'inv_freq':
        weights = (num_samples / counts) ** power
        weights = weights / weights.mean()
    else:
        if not 0. < beta < 1.:
            raise ValueError('beta must be in (0, 1).')
        weights = (1. - beta) / (1. - beta ** counts)
        weights = weights / weights.mean()
    if max_weight is not None:
        weights = weights.clamp(max=max_weight)
    return kind, weights.to(torch.float32)


def load_class_stats(path: Optional[str]) -> Optional[Tuple[torch.Tensor, int]]:
    """Load per-class positive counts and the sample count from a class_weights.py stats file.

    Args:
        path: JSON file with 'pos_counts' and 'num_samples' entries. None or an empty string returns None.

    Returns:
        The float32 counts and the number of samples they were taken over.
    """
    if not path:
        return None
    with open(path) as f:
        stats = json.load(f)
    if not isinstance(stats, dict) or 'pos_counts' not in stats or 'num_samples' not in stats:
        raise ValueError(
            f"Class stats file {path} needs 'pos_counts' and 'num_samples' entries (see class_weights.py).")
    counts = torch.as_tensor(stats['pos_counts'], dtype=torch.float32)
    num_samples = int(stats['num_samples'])
    if counts.ndim != 1 or not counts.numel() or (counts < 0).any() or (counts > num_samples).any():
        raise ValueError(f'Class stats file {path} has invalid counts for {num_samples} samples.')
    return counts, num_samples


def load_class_weights(
        value: Optional[Union[str, float, Sequence[float], torch.Tensor]],
        key: str = 'class_weight',
) -> Optional[Union[float, torch.Tensor]]:
    """Parse loss weights from a number, a comma separated list, or a file.

    Files are JSON (a list, or an object holding the list under `key`, as written by class_weights.py) or text
    with comma / whitespace separated numbers.

    Args:
        value: Weight value, list, or path. None or an empty string returns None.
        key: Entry to read from JSON objects, e.g. 'pos_weight' or 'class_weight'.

    Returns:
        A float for a single value (broadcast over classes), else a 1-D float32 tensor.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        if os.path.isfile(value):
            if value.endswith('.json'):
                with open(value) as f:
                    value = json.load(f)
                if isinstance(value, dict):
                    if key not in value:
                        raise ValueError(
                            f"Weight file has no '{key}' entry, it contains {sorted(value)}. Weights computed for "
                            "another loss argument (pos_weight vs class_weight) must be passed via that option.")
                    value = value[key]
            else:
                with open(value) as f:
                    value = f.read()
        if isinstance(value, str):
            value = [float(v) for v in value.replace(',', ' ').split()]
    if isinstance(value, bool):
        raise ValueError('Loss weights must be numbers.')
    weights = torch.as_tensor(value, dtype=torch.float32)
    if weights.ndim > 1 or not weights.numel():
        raise ValueError('Loss weights must be a single value or a non-empty 1-D list.')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Loss weights must be finite and non-negative.')
    if weights.numel() == 1:
        return float(weights.reshape(-1)[0])
    return weights


def resolve_class_weights(
        value: Optional[Union[str, float, Sequence[float], torch.Tensor]],
        kind: str = 'class_weight',
        class_stats: Optional[Tuple[torch.Tensor, int]] = None,
        power: float = 1.,
        beta: float = 0.999,
        max_weight: Optional[float] = None,
) -> Optional[Union[float, torch.Tensor]]:
    """Resolve loss weights given as a method name (computed from class stats) or as explicit weights.

    Args:
        value: A method in CLASS_WEIGHT_METHODS, or explicit weights for load_class_weights().
        kind: The loss argument being resolved, 'pos_weight' or 'class_weight'.
        class_stats: (counts, num_samples) from load_class_stats(), required for method names.
        power: See compute_class_weights().
        beta: See compute_class_weights().
        max_weight: See compute_class_weights().
    """
    if isinstance(value, str) and value.strip() in CLASS_WEIGHT_METHOD_KINDS:
        method = value.strip()
        if CLASS_WEIGHT_METHOD_KINDS[method] != kind:
            valid = [m for m, k in CLASS_WEIGHT_METHOD_KINDS.items() if k == kind]
            raise ValueError(f"Method '{method}' computes {CLASS_WEIGHT_METHOD_KINDS[method]} values, "
                             f"{kind} methods are {valid}.")
        if class_stats is None:
            raise ValueError(f"Class weight method '{method}' requires class stats (see class_weights.py).")
        _, weights = compute_class_weights(
            class_stats[0], class_stats[1], method=method, power=power, beta=beta, max_weight=max_weight)
        return weights
    return load_class_weights(value, key=kind)
