import torch

from .scheduler import Scheduler


class PlateauLRScheduler(Scheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self,
                 optimizer,
                 factor=0.1,
                 patience=10,
                 verbose=False,
                 threshold=1e-4,
                 cooldown_epochs=0,
                 warmup_updates=0,
                 warmup_lr_init=0,
                 lr_min=0,
                 ):
        super().__init__(optimizer, 'lr', initialize=False)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer,
            patience=patience,
            factor=factor,
            verbose=verbose,
            threshold=threshold,
            cooldown=cooldown_epochs,
            min_lr=lr_min
        )

        self.warmup_updates = warmup_updates
        self.warmup_lr_init = warmup_lr_init

        if self.warmup_updates:
            self.warmup_active = warmup_updates > 0  # this state updates with num_updates
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_updates for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

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
    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None and not self.warmup_active:
            self.lr_scheduler.step(val_loss, epoch)
        else:
            self.lr_scheduler.last_epoch = epoch

    def get_update_values(self, num_updates: int):
        if num_updates < self.warmup_updates:
            lrs = [self.warmup_lr_init + num_updates * s for s in self.warmup_steps]
        else:
            self.warmup_active = False  # warmup cancelled by first update past warmup_update count
            lrs = None  # no change on update after warmup stage
        return lrs

