from typing import Callable, Optional, Union, Any

import torch

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


class UpdaterXla(Updater):

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            clip_value: Optional[Union[Callable, float]] = None,
            clip_mode: str = 'norm',
            use_scaler: bool = False,
            scaler_kwargs: Any = None,
    ):
        super().__init__(optimizer=optimizer, clip_value=clip_value, clip_mode=clip_mode)
        self.after_step_closure = True
        if use_scaler:
            assert xa is not None, 'XLA AMP not present in this build'
            self.scaler = xa.GradScaler(**scaler_kwargs)

    def apply(self, loss: torch.Tensor, accumulate: bool = False):
        if self.scaler is None:
            loss.backward(create_graph=self.create_graph)
            gradients = xm._fetch_gradients(self.optimizer)
            xm.all_reduce('sum', gradients, scale=1.0 / xm.xrt_world_size())
            if self.clipper is not None:
                self.clipper()
            if not accumulate:
                xm.optimizer_step(self.optimizer)
        else:
            self.scaler.scale(loss).backward(create_graph=self.create_graph)
            if self.clipper is not None:
                self.scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                self.clipper()
            if not accumulate:
                self.scaler.step(self.optimizer)
                self.reset()
            self.scaler.update()

    def after_step(self, after_step_fn, *args):
        xm.add_step_closure(after_step_fn, *args)

