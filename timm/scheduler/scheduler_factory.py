""" Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import List, Optional, Union

from torch.optim import Optimizer

from .cosine_lr import CosineLRScheduler
from .multistep_lr import MultiStepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .poly_lr import PolyLRScheduler
from .step_lr import StepLRScheduler
from .tanh_lr import TanhLRScheduler
from .wsd_lr import WSDLRScheduler


def scheduler_kwargs(cfg, decreasing_metric: Optional[bool] = None):
    """ cfg/argparse to kwargs helper
    Convert scheduler args in argparse args or cfg (.dot) like object to keyword args.
    """
    eval_metric = getattr(cfg, 'eval_metric', 'top1')
    if decreasing_metric is not None:
        plateau_mode = 'min' if decreasing_metric else 'max'
    else:
        plateau_mode = 'min' if 'loss' in eval_metric else 'max'
    kwargs = dict(
        sched=cfg.sched,
        num_epochs=getattr(cfg, 'epochs', 100),
        decay_epochs=getattr(cfg, 'decay_epochs', 30),
        decay_milestones=getattr(cfg, 'decay_milestones', [30, 60]),
        warmup_epochs=getattr(cfg, 'warmup_epochs', 5),
        cooldown_epochs=getattr(cfg, 'cooldown_epochs', 0),
        patience_epochs=getattr(cfg, 'patience_epochs', 10),
        decay_rate=getattr(cfg, 'decay_rate', 0.1),
        min_lr=getattr(cfg, 'min_lr', 0.),
        warmup_lr=getattr(cfg, 'warmup_lr', 1e-5),
        warmup_prefix=getattr(cfg, 'warmup_prefix', False),
        noise=getattr(cfg, 'lr_noise', None),
        noise_pct=getattr(cfg, 'lr_noise_pct', 0.67),
        noise_std=getattr(cfg, 'lr_noise_std', 1.),
        noise_seed=getattr(cfg, 'seed', 42),
        cycle_mul=getattr(cfg, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(cfg, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(cfg, 'lr_cycle_limit', 1),
        k_decay=getattr(cfg, 'lr_k_decay', 1.0),
        decay_fn=getattr(cfg, 'lr_decay_fn', None),
        stable_timescale=getattr(cfg, 'lr_stable_timescale', None),
        plateau_mode=plateau_mode,
        step_on_epochs=not getattr(cfg, 'sched_on_updates', False),
    )
    return kwargs


def create_scheduler(
        args,
        optimizer: Optimizer,
        updates_per_epoch: int = 0,
):
    return create_scheduler_v2(
        optimizer=optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )


def create_scheduler_v2(
        optimizer: Optimizer,
        sched: str = 'cosine',
        num_epochs: int = 300,
        decay_epochs: Optional[float] = None,
        decay_milestones: List[int] = (90, 180, 270),
        cooldown_epochs: int = 0,
        patience_epochs: int = 10,
        decay_rate: float = 0.1,
        min_lr: float = 0.,
        warmup_lr: float = 1e-5,
        warmup_epochs: int = 0,
        warmup_prefix: bool = False,
        noise: Union[float, List[float]] = None,
        noise_pct: float = 0.67,
        noise_std: float = 1.,
        noise_seed: int = 42,
        cycle_mul: float = 1.,
        cycle_decay: float = 0.1,
        cycle_limit: int = 1,
        k_decay: float = 1.0,
        decay_fn: Optional[str] = None,
        stable_timescale: Optional[float] = None,
        plateau_mode: str = 'max',
        step_on_epochs: bool = True,
        updates_per_epoch: int = 0,
):
    """Create a learning rate scheduler by name.

    Args:
        optimizer: optimizer to schedule
        sched: scheduler name, one of 'cosine', 'tanh', 'step', 'multistep', 'plateau', 'poly', 'wsd', 'rsqrt'
        num_epochs: number of epochs in the schedule (excluding warmup if warmup_prefix, excluding cooldown)
        decay_epochs: lr decay interval (step), or the length of the decay phase (wsd / rsqrt) in epochs, a value
            in (0, 1) is a fraction of the schedule length. None = 90 (step), 0.2 (wsd / rsqrt)
        decay_milestones: epochs to decay the lr at (multistep)
        cooldown_epochs: epochs to hold the lr at min_lr after the schedule ends
        patience_epochs: epochs w/o improvement before decaying the lr (plateau)
        decay_rate: lr decay rate (step, multistep, plateau), the polynomial power (poly)
        min_lr: lr lower bound for schedulers that decay to it
        warmup_lr: lr at the start of warmup
        warmup_epochs: epochs of linear warmup
        warmup_prefix: exclude the warmup period from num_epochs (added to the total), wsd / rsqrt are always warmup
            prefixed, with warmup_prefix=False their stable + decay phases fit within num_epochs
        noise: lr noise on / off points as fractions of the schedule length
        noise_pct: lr noise limit percent
        noise_std: lr noise std-dev
        noise_seed: lr noise seed
        cycle_mul: cycle length multiplier (cosine, tanh, poly)
        cycle_decay: lr decay per cycle (cosine, tanh, poly)
        cycle_limit: number of cycles (cosine, tanh, poly)
        k_decay: k-decay (cosine, poly)
        decay_fn: decay phase function (wsd / rsqrt), None = '1-sqrt' for wsd, 'linear' for rsqrt
        stable_timescale: rsqrt timescale in epochs (rsqrt), None = warmup_epochs
        plateau_mode: 'max' or 'min', the metric direction (plateau)
        step_on_epochs: step the schedule per epoch, otherwise per update (requires updates_per_epoch)
        updates_per_epoch: number of optimizer updates per epoch (step_on_epochs=False)

    Returns:
        (scheduler, num_epochs) w/ num_epochs the total number of epochs to train including warmup prefix and cooldown
    """
    if decay_epochs is None:
        # wsd / rsqrt decay over the final 20% of the stable + decay length by default, step every 90 epochs
        decay_epochs = 0.2 if sched in ('wsd', 'rsqrt') else 90

    t_initial = num_epochs
    warmup_t = warmup_epochs
    decay_t = decay_epochs
    cooldown_t = cooldown_epochs
    timescale_t = stable_timescale

    if not step_on_epochs:
        assert updates_per_epoch > 0, 'updates_per_epoch must be set to number of dataloader batches'
        t_initial = t_initial * updates_per_epoch
        warmup_t = warmup_t * updates_per_epoch
        decay_t = decay_t * updates_per_epoch
        decay_milestones = [d * updates_per_epoch for d in decay_milestones]
        cooldown_t = cooldown_t * updates_per_epoch
        if timescale_t is not None:
            timescale_t = timescale_t * updates_per_epoch

    # warmup args
    warmup_args = dict(
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_t,
        warmup_prefix=warmup_prefix,
    )

    # setup noise args for supporting schedulers
    if noise is not None:
        if isinstance(noise, (list, tuple)):
            noise_range = [n * t_initial for n in noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = noise * t_initial
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=noise_pct,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )

    # setup cycle args for supporting schedulers
    cycle_args = dict(
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
        cycle_limit=cycle_limit,
    )

    lr_scheduler = None
    if sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_t,
            decay_rate=decay_rate,
            t_in_epochs=step_on_epochs,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_milestones,
            decay_rate=decay_rate,
            t_in_epochs=step_on_epochs,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'plateau':
        assert step_on_epochs, 'Plateau LR only supports step per epoch.'
        warmup_args.pop('warmup_prefix', False)
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=decay_rate,
            patience_t=patience_epochs,
            cooldown_t=0,
            **warmup_args,
            lr_min=min_lr,
            mode=plateau_mode,
            **noise_args,
        )
    elif sched == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            k_decay=k_decay,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )
    elif sched in ('wsd', 'rsqrt'):
        # Warmup is always a prefix for this scheduler, honour warmup_prefix=False (the default) by fitting the
        # stable + decay phases within num_epochs so --epochs means the same as for the other schedulers.
        t_sched = t_initial if warmup_prefix else t_initial - warmup_t
        if 0 < decay_epochs < 1:
            decay_t = decay_epochs * t_sched  # fraction of the stable + decay length
        decay_t = int(round(decay_t))
        if not 0 < decay_t <= t_sched:
            raise ValueError(
                f'{sched} schedule requires 0 < decay_epochs <= epochs{"" if warmup_prefix else " - warmup_epochs"}, '
                f'got decay {decay_t} and stable + decay length {t_sched}.')
        lr_scheduler = WSDLRScheduler(
            optimizer,
            t_initial=t_sched,
            decay_t=decay_t,
            lr_min=min_lr,
            stable_fn='rsqrt' if sched == 'rsqrt' else 'constant',
            stable_timescale=timescale_t,
            decay_fn=decay_fn or ('linear' if sched == 'rsqrt' else '1-sqrt'),
            t_in_epochs=step_on_epochs,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr,
            **noise_args,
        )

    if hasattr(lr_scheduler, 'get_cycle_length'):
        # For cycle based schedulers (cosine, tanh, poly) recalculate total epochs w/ cycles & cooldown
        # NOTE: Warmup prefix added in get_cycle_lengths() if enabled
        t_with_cycles_and_cooldown = lr_scheduler.get_cycle_length() + cooldown_t
        if step_on_epochs:
            num_epochs = t_with_cycles_and_cooldown
        else:
            num_epochs = t_with_cycles_and_cooldown // updates_per_epoch
    else:
        if warmup_prefix:
            num_epochs += warmup_epochs

    return lr_scheduler, num_epochs
