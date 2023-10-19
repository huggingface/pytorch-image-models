import functools
import types
from typing import Any, Dict, List, Tuple, Union

import torch.nn


BlockArgs = List[List[Dict[str, Any]]]
LayerType = Union[type, str, types.FunctionType, functools.partial, torch.nn.Module]
PadType = Union[str, int, Tuple[int, int]]
