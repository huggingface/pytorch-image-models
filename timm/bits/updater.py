from dataclasses import dataclass, field, InitVar
from functools import partial
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from .grad_clip import get_clip_grad_fn, get_clip_parameters


@dataclass
class Updater:

    model: nn.Module = None
    optimizer: torch.optim.Optimizer = None  # FIXME handle multiple optimizers per-model
    clip_fn: Optional[Union[Callable, str]] = None
    clip_value: Optional[float] = None
    clip_params_fn: Optional[Callable] = None
    grad_scaler: Optional[Callable] = None
    create_graph: Optional[bool] = None
    after_step_closure: bool = False

    def __post_init__(self):
        assert self.model is not None
        assert self.optimizer is not None
        if self.clip_fn is not None:
            if isinstance(self.clip_fn, Callable):
                skip_last = 0
            else:
                assert isinstance(self.clip_fn, str)
                skip_last = 2 if 'agc' in self.clip_fn else 0
                self.clip_fn = get_clip_grad_fn(self.clip_fn)
                assert self.clip_value is not None
            self.clip_params_fn = partial(get_clip_parameters, model=self.model, skip_last=skip_last)
        if self.create_graph is None:
            self.create_graph = getattr(self.optimizer, 'second_order', False)
        self.after_step_closure = False

    def reset(self):
        self.optimizer.zero_grad()

    def apply(self, loss: torch.Tensor, accumulate=False):
        loss.backward(create_graph=self.create_graph)
        if accumulate:
            return
        if self.clip_fn is not None:
            self.clip_fn(self.clip_params_fn(), self.clip_value)
        self.optimizer.step()
        self.reset()

    def get_average_lr(self):
        lrl = [param_group['lr'] for param_group in self.optimizer.param_groups if param_group['lr'] > 0]
        return sum(lrl) / len(lrl)

    def state_dict(self):
        state_dict = dict(optimizer=self.optimizer.state_dict())
        if self.grad_scaler is not None:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'grad_scaler' in state_dict and self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(state_dict['grad_scaler'])

    def after_step(self, after_step_fn, *args):
        after_step_fn(*args)

    @property
    def deepspeed(self):
        return False
