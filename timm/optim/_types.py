from typing import Any, Dict, Iterable, Union, Protocol, Type
try:
    from typing import TypeAlias, TypeVar
except ImportError:
    from typing_extensions import TypeAlias, TypeVar

import torch
import torch.optim

try:
    from torch.optim.optimizer import ParamsT
except (ImportError, TypeError):
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


OptimType = Type[torch.optim.Optimizer]


class OptimizerCallable(Protocol):
    """Protocol for optimizer constructor signatures."""

    def __call__(self, params: ParamsT, **kwargs) -> torch.optim.Optimizer: ...


__all__ = ['ParamsT', 'OptimType', 'OptimizerCallable']