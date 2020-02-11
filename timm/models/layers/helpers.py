""" Layer/Module Helpers

Hacked together by Ross Wightman
"""
from itertools import repeat
from torch._six import container_abcs


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


tup_single = _ntuple(1)
tup_pair = _ntuple(2)
tup_triple = _ntuple(3)
tup_quadruple = _ntuple(4)






