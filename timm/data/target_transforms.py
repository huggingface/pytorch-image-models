"""Transforms for classification targets."""
import torch


class MultiLabelTarget:
    """Encode a multi-label target as a float32 multi-hot tensor of length num_classes.

    By default the target is a sequence of positive class indices: an empty sequence
    means no positive labels, repeats are allowed, and negative, out-of-range, or
    non-integer values are rejected. With dense=True the target is already a vector
    of length num_classes holding bool, int, or float values in [0, 1]; it is validated
    and cast to float32, keeping soft values.
    """

    def __init__(self, num_classes: int, dense: bool = False):
        if num_classes <= 0:
            raise ValueError('num_classes must be positive.')
        self.num_classes = num_classes
        self.dense = dense

    def __call__(self, target) -> torch.Tensor:
        values = torch.as_tensor(target)
        if values.ndim != 1:
            raise ValueError('Multi-label targets must be a 1-D sequence of class indices or a dense target vector.')
        if self.dense:
            if values.numel() != self.num_classes:
                raise ValueError(
                    f'Dense multi-label targets must have {self.num_classes} entries, got {values.numel()}.')
            if values.is_complex():
                raise ValueError('Dense multi-label target values must be real numbers.')
            values = values.to(torch.float32)
            if not ((values >= 0) & (values <= 1)).all():  # also rejects NaN
                raise ValueError('Dense multi-label target values must be in [0, 1].')
            return values
        result = torch.zeros(self.num_classes, dtype=torch.float32)
        if not values.numel():
            return result
        if values.is_floating_point() or values.is_complex() or values.dtype == torch.bool:
            raise ValueError('Multi-label class indices must be integers (use dense=True for multi-hot vectors).')
        if (values < 0).any() or (values >= self.num_classes).any():
            raise ValueError(f'Multi-label class indices must be in [0, {self.num_classes}).')
        result[values.long()] = 1.
        return result
