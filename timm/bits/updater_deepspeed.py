from dataclasses import dataclass, field, InitVar

import torch
try:
    import deepspeed as ds
except ImportError as e:
    ds = None

from .updater import Updater


@dataclass
class UpdaterDeepSpeed(Updater):

    def __post_init__(self):
        super().__post_init__()
        # FIXME not sure how to deal with model.module / grad clipping w/ DS engine interface
        assert isinstance(self.model, ds.DeepSpeedEngine)

    def reset(self):
        self.model.zero_grad()

    def apply(self, loss: torch.Tensor, accumulate=False):
        self.model.backward(loss)
        self.model.step()
        self.reset()

    @property
    def deepspeed(self):
        return True
