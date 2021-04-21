from typing import Callable, Optional, Union

import torch

from .grad_clipper import GradClipper


class Updater:

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            clip_value: Optional[Union[Callable, float]] = None,
            clip_mode: str = 'norm'):

        self.optimizer = optimizer
        self.clipper: Optional[GradClipper] = None
        if clip_value is not None:
            if isinstance(clip_value, Callable):
                self.clipper = clip_value
            else:
                GradClipper(clip_value, clip_mode)
        self.scaler = None
        self.create_graph = getattr(self.optimizer, 'second_order', False)
        self.num_accumulated = 0
        self.after_step_closure = False

    def apply(self, loss: torch.Tensor, accumulate=False):
        loss.backward(create_graph=self.create_graph)
        if self.clipper is not None:
            self.clipper()
        if not accumulate:
            self.optimizer.step()
            self.reset()
        else:
            self.num_accumulated += 1

    def reset(self):
        self.optimizer.zero_grad()
        self.num_accumulated = 0

    def state_dict(self):
        state_dict = dict(optimizer=self.optimizer.state_dict())
        if self.scaler is not None:
            state_dict['scaler'] = self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'scaler' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])



