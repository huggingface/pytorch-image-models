""" Step Scheduler

Basic step LR schedule with warmup, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch
from typing import List, Tuple, Union


from .scheduler import Scheduler


class StepLRScheduler(Scheduler):
    """
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            decay_t: float,
            decay_rate: float = 1.,
            warmup_t: int = 0,
            warmup_lr_init: float = 0.,
            warmup_prefix: bool = True,
            t_in_epochs: bool = True,
            noise_range_t: Union[List[int], Tuple[int, int], int, None] = None,
            noise_pct: float = 0.67,
            noise_std: float = 1.0,
            noise_seed: int = 42,
            initialize: bool = True,
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

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t: int) -> List[float]:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t
            lrs = [v * (self.decay_rate ** (t // self.decay_t)) for v in self.base_values]
        return lrs
