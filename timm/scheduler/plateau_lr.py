""" Plateau Scheduler

Adapts PyTorch plateau scheduler and allows application of noise, warmup.

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

from .scheduler import Scheduler


class PlateauLRScheduler(Scheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self,
                 optimizer,
                 decay_rate=0.1,
                 patience_t=10,
                 verbose=True,
                 threshold=1e-4,
                 cooldown_t=0,
                 warmup_t=0,
                 warmup_lr_init=0,
                 lr_min=0,
                 mode='max',
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize=True,
                 ):
        super().__init__(optimizer, 'lr', initialize=initialize)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=patience_t,
            factor=decay_rate,
            verbose=verbose,
            threshold=threshold,
            cooldown=cooldown_t,
            mode=mode,
            min_lr=lr_min
        )

        self.noise_range = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        self.restore_lr = None

    def state_dict(self):
        return {
            'best': self.lr_scheduler.best,
            'last_epoch': self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.lr_scheduler.best = state_dict['best']
        if 'last_epoch' in state_dict:
            self.lr_scheduler.last_epoch = state_dict['last_epoch']

    # override the base class step fn completely
    def step(self, epoch, metric=None):
        if epoch <= self.warmup_t:
            lrs = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
            super().update_groups(lrs)
        else:
            if self.restore_lr is not None:
                # restore actual LR from before our last noise perturbation before stepping base
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.restore_lr[i]
                self.restore_lr = None

            self.lr_scheduler.step(metric, epoch)  # step the base scheduler

            if self.noise_range is not None:
                if isinstance(self.noise_range, (list, tuple)):
                    apply_noise = self.noise_range[0] <= epoch < self.noise_range[1]
                else:
                    apply_noise = epoch >= self.noise_range
                if apply_noise:
                    self._apply_noise(epoch)

    def _apply_noise(self, epoch):
        g = torch.Generator()
        g.manual_seed(self.noise_seed + epoch)
        if self.noise_type == 'normal':
            while True:
                # resample if noise out of percent limit, brute force but shouldn't spin much
                noise = torch.randn(1, generator=g).item()
                if abs(noise) < self.noise_pct:
                    break
        else:
            noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct

        # apply the noise on top of previous LR, cache the old value so we can restore for normal
        # stepping of base scheduler
        restore_lr = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            restore_lr.append(old_lr)
            new_lr = old_lr + old_lr * noise
            param_group['lr'] = new_lr
        self.restore_lr = restore_lr
