""" Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise, k-decay.

Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
import math
from typing import List, Union
import torch

from .scheduler import Scheduler


_logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909

    Args:
        optimizer (torch.optim.Optimizer): torch optimizer to schedule  
        t_initial (int): Number of epochs it initial (first) cycle.  
        lr_min (float, optional):  Minimum learning rate to use during the scheduling. Defaults to 0..  
        cycle_mul (float, optional): Multiplyer for cycle length. Defaults to 1..  
        cycle_decay (float, optional): Factor to decay lr at next cycle. Defaults to 1..  
        cycle_limit (int, optional): Number of cycles. Defaults to 1.  
        warmup_t (int, optional): Number of epochs to warmup. Defaults to 0.  
        warmup_lr_init (float, optional): Initial learning rate during warmup . Defaults to 0.  
        warmup_prefix (bool, optional): If True, after warmup annealing starts from initial LR. Defaults to False.  
        t_in_epochs (bool, optional): If set to False, returned lr are None. Defaults to True.  
        noise_range_t (Union[int, float, List[int | float]], optional): Epoch when noise starts.\
            If list or tuple - epoch range, when noise applied. Defaults to None.  
        noise_pct (float, optional): Percentage of noise to add. Defaults to 0.67.  
        noise_std (float, optional): Noise standard deviation. Defaults to 1.0.  
        noise_seed (int, optional): Seed to use to add random noise. Defaults to 42.  
        k_decay (float, optional): Power for k_decay. Defaults to 1.0.  
        initialize (bool, optional): Add initial_{field_name} to optimizer param group. Defaults to True.  
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min: float = 0.,
                 cycle_mul: float = 1.,
                 cycle_decay: float = 1.,
                 cycle_limit: int = 1,
                 warmup_t: int = 0,
                 warmup_lr_init: float = 0,
                 warmup_prefix: bool = False,
                 t_in_epochs: bool = True,
                 noise_range_t: Union[int, float, List[Union[int, float]]] = None,
                 noise_pct: float = 0.67,
                 noise_std: float = 1.0,
                 noise_seed: int = 42,
                 k_decay: float = 1.0,
                 initialize: bool = True) -> None:
        
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
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

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles: int = 0) -> int:
        """Return total number of epochs.

        Args:
            cycles (int, optional): Number of cycles. If 0, takes cycle_limit from sched. Defaults to 0.

        Returns:
            int: Total number of epochs
        """
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))
