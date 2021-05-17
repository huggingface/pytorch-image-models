from dataclasses import dataclass
from typing import Dict, Any

from torch import nn as nn

from timm.scheduler import Scheduler
from .updater import Updater


@dataclass
class TrainState:
    model: nn.Module = None
    train_loss: nn.Module = None
    eval_loss: nn.Module = None
    updater: Updater = None
    lr_scheduler: Scheduler = None
    model_ema: nn.Module = None

    step_count_epoch: int = 0
    step_count_global: int = 0
    epoch: int = 0

    def __post_init__(self):
        assert self.model is not None
        assert self.updater is not None


def serialize_train_state(train_state: TrainState):
    pass


def deserialize_train_state(checkpoint: Dict[str, Any]):
    pass