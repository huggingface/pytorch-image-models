""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    """Return a function that converts input to an n-tuple.

    Scalar values are repeated n times, while iterables are converted to tuples.
    Strings are treated as scalars to avoid character-level splitting.

    Args:
        n: Target tuple length.

    Returns:
        Function that converts input to n-tuple.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    """Adjust value to be divisible by a divisor, typically for channel counts.

    Rounds to the nearest multiple of divisor while ensuring the result doesn't
    fall below min_value or decrease by more than (1 - round_limit).

    Args:
        v: Value to adjust.
        divisor: Target divisor.
        min_value: Minimum acceptable value.
        round_limit: Prevent decrease beyond this fraction of original value.

    Returns:
        Adjusted value divisible by divisor.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def extend_tuple(x, n):
    """Pad a tuple to length n by repeating the last value.

    If input is shorter than n, extends by repeating the last element.
    If input is longer than n, truncates to n.

    Args:
        x: Input value, tuple, or list.
        n: Target length.

    Returns:
        Tuple of length n.
    """
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n
