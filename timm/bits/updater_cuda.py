from typing import Callable, Optional, Union, Any

import torch

from .updater import Updater


class UpdaterCuda(Updater):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            clip_value: Optional[Union[Callable, float]] = None,
            clip_mode: str = 'norm',
            use_scaler: bool = False,
            scaler_kwargs: Any = None,
    ):
        super().__init__(optimizer=optimizer, clip_value=clip_value, clip_mode=clip_mode)
        scaler_kwargs = scaler_kwargs or {}
        if use_scaler:
            self.scaler = torch.cuda.amp.GradScaler(**scaler_kwargs)

    def apply(self, loss: torch.Tensor, accumulate=False):
        if self.scaler is not None:
            self.scaler.scale(loss).backward(create_graph=self.create_graph)
            if self.clipper is not None:
                self.scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                self.clipper()
            if not accumulate:
                self.scaler.step(self.optimizer)
                self.reset()
            else:
                self.num_accumulated += 1
            self.scaler.update()
        else:
            Updater.apply(self, loss, accumulate)

