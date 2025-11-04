import torch
from torch import nn


class LayerScale(nn.Module):
    """ LayerScale on tensors with channels in last-dim.
    """
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        self.init_values = init_values
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale2d(nn.Module):
    """ LayerScale for tensors with torch 2D NCHW layout.
    """
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.init_values = init_values
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma

