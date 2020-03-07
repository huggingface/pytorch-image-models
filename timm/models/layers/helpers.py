""" Layer/Module Helpers

Hacked together by Ross Wightman
"""
from itertools import repeat
from torch._six import container_abcs


# From PyTorch internals
def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


tup_single = ntuple(1)
tup_pair = ntuple(2)
tup_triple = ntuple(3)
tup_quadruple = ntuple(4)


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



