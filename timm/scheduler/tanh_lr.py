""" TanH Scheduler

TanH schedule with warmup, cycle/restarts, noise.

Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
import math
import numpy as np
import torch
from typing import List

from .scheduler import Scheduler


_logger = logging.getLogger(__name__)


class TanhLRScheduler(Scheduler):
    """
    Hyberbolic-Tangent decay with restarts.
    This is described in the paper https://arxiv.org/abs/1806.01593
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lb: float = -7.,
            ub: float = 3.,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        assert lb < ub
        assert cycle_limit >= 0
        assert warmup_t >= 0
        assert warmup_lr_init >= 0
        self.lb = lb
        self.ub = ub
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        if self.warmup_t:
            t_v = self.base_values if self.warmup_prefix else self._get_lr(self.warmup_t)
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in t_v]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t: int) -> List[float]:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            if i < self.cycle_limit:
                gamma = self.cycle_decay ** i
                lr_max_values = [v * gamma for v in self.base_values]

                tr = t_curr / t_i
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 - math.tanh(self.lb * (1. - tr) + self.ub * tr))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]
        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            t = self.t_initial * cycles
        else:
            t = int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))
        return t + self.warmup_t if self.warmup_prefix else t
