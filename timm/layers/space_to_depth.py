import torch
import torch.nn as nn


class SpaceToDepth(nn.Module):
    """Rearrange spatial dimensions into channel dimension.

    Divides spatial dimensions by block_size and multiplies channels by block_size^2.
    Used in TResNet as an efficient stem operation.

    Args:
        block_size: Spatial reduction factor.
    """
    bs: torch.jit.Final[int]

    def __init__(self, block_size: int = 4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * self.bs * self.bs, H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class DepthToSpace(nn.Module):
    """Rearrange channel dimension into spatial dimensions.

    Inverse of SpaceToDepth. Divides channels by block_size^2 and multiplies
    spatial dimensions by block_size.

    Args:
        block_size: Spatial expansion factor.
    """

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x
