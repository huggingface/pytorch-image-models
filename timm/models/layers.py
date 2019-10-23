import torch
import torch.nn as nn
import torch.nn.functional as F


def versiontuple(v):
    return tuple(map(int, (v.split("."))))[:3]


if versiontuple(torch.__version__) >= versiontuple('1.2.0'):
    Flatten = nn.Flatten
else:
    class Flatten(nn.Module):
        r"""
        Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
        Args:
            start_dim: first dim to flatten (default = 1).
            end_dim: last dim to flatten (default = -1).
        Shape:
            - Input: :math:`(N, *dims)`
            - Output: :math:`(N, \prod *dims)` (for the default case).
        """
        __constants__ = ['start_dim', 'end_dim']

        def __init__(self, start_dim=1, end_dim=-1):
            super(Flatten, self).__init__()
            self.start_dim = start_dim
            self.end_dim = end_dim

        def forward(self, input):
            return input.flatten(self.start_dim, self.end_dim)
