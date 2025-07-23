from contextlib import nullcontext
from functools import wraps
from typing import Callable, Optional, Tuple, Type, TypeVar, Union, overload, ContextManager

import torch

__all__ = ["LayerType", "PadType", "nullwrap", "disable_compiler"]


LayerType = Union[str, Callable, Type[torch.nn.Module]]
PadType = Union[str, int, Tuple[int, int]]

F = TypeVar("F", bound=Callable[..., object])


@overload
def nullwrap(fn: F) -> F: ...  # decorator form

@overload
def nullwrap(fn: None = ...) -> ContextManager: ...  # context‑manager form

def nullwrap(fn: Optional[F] = None):
    # as a context manager
    if fn is None:
        return nullcontext()  # `with nullwrap():`

    # as a decorator
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper  # `@nullwrap`


disable_compiler = getattr(getattr(torch, "compiler", None), "disable", None) or nullwrap
