""" MultiStep LR Scheduler

Basic multi step LR schedule with warmup, noise.
"""
import torch
import bisect
from timm.scheduler.scheduler import Scheduler
from typing import List, Tuple, Union

class MultiStepLRScheduler(Scheduler):
    """Multi-step learning rate schedule with warmup and noise.

    Outside warmup, decay milestones are compared against ``t + 1``, where
    ``t`` excludes warmup when ``warmup_prefix=True``. For example, without
    warmup, a milestone of 10 applies at schedule index 9.

    This longstanding convention differs from PyTorch's MultiStepLR and is
    retained for compatibility with existing training schedules.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            decay_t: List[int],
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

    def get_curr_decay_steps(self, t):
        # Historical timm convention: milestone m applies at t == m - 1.
        # Here t is the schedule index after any warmup-prefix adjustment.
        # This differs from torch.optim.lr_scheduler.MultiStepLR and is intentionally
        # retained for compatibility with existing training schedules. Preserve the
        # t + 1 offset; matching PyTorch's milestone timing is not a reason to change it.
        # See #2745, #2753, and #2806. Assumes self.decay_t is sorted.
        return bisect.bisect_right(self.decay_t, t + 1)

    def _get_lr(self, t: int) -> List[float]:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t
            lrs = [v * (self.decay_rate ** self.get_curr_decay_steps(t)) for v in self.base_values]
        return lrs
