from dataclasses import dataclass, field, InitVar
from typing import Dict, Any

import torch

from .updater import Updater


@dataclass
class UpdaterCudaWithScaler(Updater):

    scaler_kwargs: InitVar[Dict[str, Any]] = None

    def __post_init__(self, scaler_kwargs: Dict[str, Any]):
        super().__post_init__()
        scaler_kwargs = scaler_kwargs or {}
        self.grad_scaler = torch.cuda.amp.GradScaler(**scaler_kwargs)

    def apply(self, loss: torch.Tensor, accumulate=False):
        self.grad_scaler.scale(loss).backward(create_graph=self.create_graph)
        if accumulate:
            # unscale first?
            return
        if self.clip_fn is not None:
            # unscale the gradients of optimizer's assigned params in-place
            self.grad_scaler.unscale_(self.optimizer)
            self.clip_fn(self.clip_params_fn(), self.clip_value)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.reset()
