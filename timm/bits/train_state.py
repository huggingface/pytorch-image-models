from dataclasses import dataclass
from typing import Dict, Any

from torch import nn as nn

from timm.scheduler import Scheduler
from timm.utils import get_state_dict, unwrap_model

from .updater import Updater


@dataclass
class TrainState:
    model: nn.Module = None
    train_loss: nn.Module = None
    eval_loss: nn.Module = None
    updater: Updater = None
    lr_scheduler: Scheduler = None
    model_ema: nn.Module = None

    epoch: int = 0
    step_count: int = 0
    step_count_global: int = 0

    def __post_init__(self):
        assert self.model is not None
        assert self.updater is not None

    def state_dict(self, unwrap_fn=unwrap_model):
        state = dict(
            epoch=self.epoch,
            step_count=self.step_count,
            step_count_global=self.step_count_global,
            model=get_state_dict(self.model, unwrap_fn),
            model_ema=None if self.model_ema is None else get_state_dict(self.model_ema, unwrap_fn),
        )
        # FIXME lr_scheduler state save?
        state.update(self.updater.state_dict())
        return state

    def load_state_dict(self, state_dict, unwrap_fn=unwrap_model):
        self.epoch = state_dict['epoch']
        self.step_count = state_dict['step_count']
        self.step_count_global = state_dict['step_count_global']

        unwrap_fn(self.model).load_state_dict(state_dict.get('model'))
        if 'model_ema' in state_dict and self.model_ema is not None:
            unwrap_fn(self.model_ema).load_state_dict(state_dict.get('model_ema'))

        self.updater.load_state_dict(state_dict)
