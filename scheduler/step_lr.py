import math
import torch

from .scheduler import Scheduler


class StepLRScheduler(Scheduler):
    """
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_epochs: int,
                 decay_rate: float = 1.,
                 warmup_updates=0,
                 warmup_lr_init=0,
                 initialize=True) -> None:
        super().__init__(optimizer, param_group_field="lr", initialize=initialize)

        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.warmup_updates = warmup_updates
        self.warmup_lr_init = warmup_lr_init

        if self.warmup_updates:
            self.warmup_active = warmup_updates > 0  # this state updates with num_updates
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_updates for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def get_epoch_values(self, epoch: int):
        if not self.warmup_active:
            lrs = [v * (self.decay_rate ** ((epoch + 1) // self.decay_epochs))
                   for v in self.base_values]
        else:
            lrs = None  # no epoch updates while warming up
        return lrs

    def get_update_values(self, num_updates: int):
        if num_updates < self.warmup_updates:
            lrs = [self.warmup_lr_init + num_updates * s for s in self.warmup_steps]
        else:
            self.warmup_active = False  # warmup cancelled by first update past warmup_update count
            lrs = None  # no change on update afte warmup stage
        return lrs


