import functools
import types
from typing import Tuple, Union

import torch.nn


LayerType = Union[type, str, types.FunctionType, functools.partial, torch.nn.Module]
PadType = Union[str, int, Tuple[int, int]]
