from typing import Callable, Tuple, Type, Union

import torch


LayerType = Union[str, Callable, Type[torch.nn.Module]]
PadType = Union[str, int, Tuple[int, int]]
