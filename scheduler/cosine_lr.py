import logging
import math
import numpy as np
import torch

from .scheduler import Scheduler


logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    """
    Cosine annealing with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_updates=0,
                 warmup_lr_init=0,
                 initialize=True) -> None:
        super().__init__(optimizer, param_group_field="lr", initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.warmup_updates = warmup_updates
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_updates:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_updates for v in self.base_values]
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        if self.warmup_lr_init:
            super().update_groups(self.warmup_lr_init)

    def get_epoch_values(self, epoch: int):
        # this scheduler doesn't update on epoch
        return None

    def get_update_values(self, num_updates: int):
        if num_updates < self.warmup_updates:
            lrs = [self.warmup_lr_init + num_updates * s for s in self.warmup_steps]
        else:
            curr_updates = num_updates - self.warmup_updates
            if self.t_mul != 1:
                i = math.floor(math.log(1 - curr_updates / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = curr_updates - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = curr_updates // self.t_initial
                t_i = self.t_initial
                t_curr = curr_updates - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            lrs = [
                lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
            ]
        return lrs
