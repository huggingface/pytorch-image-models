""" MultiStep LR Scheduler

Basic multi step LR schedule with warmup, noise.
"""
import torch
import bisect
from timm.scheduler.scheduler import Scheduler
from typing import List

class MultiStepLRScheduler(Scheduler):
    """
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            decay_t: List[int],
            decay_rate: float = 1.,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=True,
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

    def get_curr_decay_steps(self, t):
        # find where in the array t goes,
        # assumes self.decay_t is sorted
        return bisect.bisect_right(self.decay_t, t + 1)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t
            lrs = [v * (self.decay_rate ** self.get_curr_decay_steps(t)) for v in self.base_values]
        return lrs
