from dataclasses import dataclass, field, InitVar
from typing import Any, Dict

import torch
import torch.nn as nn

try:
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except ImportError as e:
    xm = None
    _HAS_XLA = False

try:
    # only the very latest XLA builds have AMP
    import torch_xla.amp as xa
except ImportError as e:
    xa = None

from .updater import Updater


@dataclass
class UpdaterXla(Updater):

    def __post_init__(self):
        super().__post_init__()
        self.after_step_closure = True

    def apply(self, loss: torch.Tensor, accumulate: bool = False):
        loss.backward(create_graph=self.create_graph)
        if accumulate:
            return
        xm.reduce_gradients(self.optimizer)
        if self.clip_fn is not None:
            self.clip_fn(self.clip_params_fn(), self.clip_value)
        self.optimizer.step()
        xm.mark_step()
        self.reset()

    def after_step(self, after_step_fn, *args):
        xm.add_step_closure(after_step_fn, args)


@dataclass
class UpdaterXlaWithScaler(UpdaterXla):

    scaler_kwargs: InitVar[Dict[str, Any]] = None

    def __post_init__(self, scaler_kwargs: Dict[str, Any]):
        super().__post_init__()
        scaler_kwargs = scaler_kwargs or {}
        assert xa is not None, 'XLA AMP not present in this build'
        self.scaler = xa.GradScaler(**scaler_kwargs)

    def apply(self, loss: torch.Tensor, accumulate: bool = False):
        self.scaler.scale(loss).backward(create_graph=self.create_graph)
        if accumulate:
            # unscale first?
            return
        xm.reduce_gradients(self.optimizer)
        if self.clip_fn is not None:
            self.scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
            self.clip_fn(self.clip_params_fn(), self.clip_value)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        xm.mark_step()
        self.reset()
