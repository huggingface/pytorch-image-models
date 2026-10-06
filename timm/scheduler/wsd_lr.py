""" Warmup-Stable-Decay (WSD) Scheduler

Warmup -> stable phase -> decay to lr_min over the final steps, w/ selectable stable and decay functions.

References:
    MiniCPM (WSD): https://arxiv.org/abs/2404.06395
    Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (1-sqrt cooldown):
        https://arxiv.org/abs/2405.18392
    Scaling Vision Transformers (rsqrt w/ linear cooldown): https://arxiv.org/abs/2106.04560
    Power Scheduler: https://arxiv.org/abs/2408.13359

Hacked together by / Copyright 2026 Ross Wightman
"""
import math
from typing import Callable, List, Optional, Tuple, Union

import torch

from .scheduler import Scheduler


_STABLE_FNS = ('constant', 'rsqrt', 'power')
_DECAY_FNS = ('linear', '1-sqrt', 'cosine', 'power', 'exp')


class WSDLRScheduler(Scheduler):
    """ Warmup-Stable-Decay (WSD) LR schedule w/ selectable stable and decay functions.

    Three phases: a linear warmup over `warmup_t` steps, a stable phase, and a decay to `lr_min` over the final
    `decay_t` steps. `t_initial` is the length of the stable + decay phases, the warmup is always a prefix so the
    schedule has a total length of `warmup_t + t_initial` (see `get_cycle_length()`), after which `lr_min` is held.

    NOTE: as for the other schedulers, `lr_min` is reached at `t = warmup_t + t_initial`, one step past the last
    scheduled step, so the last scheduled step is one decay step short of it. With a short decay phase stepped per
    epoch this is significant (e.g. the last of 5 decay epochs w/ '1-sqrt' is at ~10% of the stable lr). The final
    anneal matters for WSD, so step per update (`t_in_epochs=False`, `--sched-on-updates` in train.py) or add a
    cooldown epoch (held at `lr_min`, `--cooldown-epochs`).

    The stable function is evaluated on the time since the end of warmup (`t_s`) and equals 1 at `t_s = 0`, so the
    warmup always hands off at the base lr. The decay function maps the decay progress `frac` in [0, 1] to a factor
    from 1 to 0 and multiplies the stable value, so a decaying stable function keeps decaying through the decay
    phase (as in the Big Vision rsqrt + linear cooldown schedule).

        lr(t) = lr_min + (base_lr * stable(t_s) - lr_min) * decay(frac)

    Stable functions:
        * `constant`: 1, the WSD schedule
        * `rsqrt`: (1 + t_s / timescale) ** -0.5, the Big Vision reciprocal square root schedule when
          `stable_timescale == warmup_t` (the default)
        * `power`: (1 + t_s / timescale) ** -stable_power, rsqrt generalized (Power scheduler)

    Decay functions:
        * `1-sqrt`: 1 - sqrt(frac), the best performing cooldown in https://arxiv.org/abs/2405.18392 (default)
        * `linear`: 1 - frac, the 'trapezoidal' schedule and the Big Vision cooldown
        * `cosine`: 0.5 * (1 + cos(pi * frac))
        * `power`: 1 - frac ** decay_power, 0.5 == '1-sqrt', 1.0 == 'linear', 2.0 is a 'square' decay
        * `exp`: exponential decay to a fraction `decay_ratio` of the stable value, normalized to reach lr_min
        * a callable `fn(frac) -> factor` for anything else

    Args:
        optimizer: optimizer to schedule
        t_initial: length of the stable + decay phases (epochs or updates, see `t_in_epochs`)
        decay_t: length of the decay phase, the final `decay_t` steps of `t_initial`
        lr_min: lr at the end of the decay phase
        stable_fn: stable phase function, one of 'constant', 'rsqrt', 'power'
        stable_timescale: timescale of the 'rsqrt' / 'power' stable functions, defaults to `warmup_t`
        stable_power: exponent of the 'power' stable function
        decay_fn: decay phase function, one of 'linear', '1-sqrt', 'cosine', 'power', 'exp' or a callable
        decay_power: exponent of the 'power' decay function
        decay_ratio: final fraction (before normalization) of the 'exp' decay function
        warmup_t: warmup length, a prefix to `t_initial`
        warmup_lr_init: lr at the start of warmup
        t_in_epochs: `t` is in epochs (`step()`), otherwise updates (`step_update()`)
        noise_range_t: lr noise range, see `Scheduler`
        noise_pct: lr noise limit percent
        noise_std: lr noise std-dev
        noise_seed: lr noise seed
        initialize: initialize the `initial_lr` param group field from the current lr
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            decay_t: int,
            lr_min: float = 0.,
            stable_fn: str = 'constant',
            stable_timescale: Optional[float] = None,
            stable_power: float = 0.5,
            decay_fn: Union[str, Callable[[float], float]] = '1-sqrt',
            decay_power: float = 0.5,
            decay_ratio: float = 0.1,
            warmup_t: int = 0,
            warmup_lr_init: float = 0.,
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

        if t_initial <= 0:
            raise ValueError(f'Invalid t_initial ({t_initial}), must be > 0.')
        if not 0 < decay_t <= t_initial:
            raise ValueError(f'Invalid decay_t ({decay_t}), must be in (0, t_initial] w/ t_initial={t_initial}.')
        if lr_min < 0:
            raise ValueError(f'Invalid lr_min ({lr_min}), must be >= 0.')
        if stable_fn not in _STABLE_FNS:
            raise ValueError(f'Invalid stable_fn ({stable_fn}), must be one of {_STABLE_FNS}.')
        if isinstance(decay_fn, str) and decay_fn not in _DECAY_FNS:
            raise ValueError(f'Invalid decay_fn ({decay_fn}), must be one of {_DECAY_FNS} or a callable.')
        if not 0. < decay_ratio < 1.:
            raise ValueError(f'Invalid decay_ratio ({decay_ratio}), must be in (0, 1).')
        if decay_power <= 0:
            raise ValueError(f'Invalid decay_power ({decay_power}), must be > 0.')
        if stable_power <= 0:
            raise ValueError(f'Invalid stable_power ({stable_power}), must be > 0.')
        if stable_fn != 'constant':
            if stable_timescale is None:
                stable_timescale = warmup_t
            if stable_timescale <= 0:
                raise ValueError(
                    f'stable_fn={stable_fn} requires a positive stable_timescale, it defaults to warmup_t '
                    f'({warmup_t}), set it explicitly when there is no warmup.')

        self.t_initial = t_initial
        self.decay_t = decay_t
        self.lr_min = lr_min
        self.stable_fn = stable_fn
        self.stable_timescale = stable_timescale
        self.stable_power = stable_power
        self.decay_fn = decay_fn
        self.decay_power = decay_power
        self.decay_ratio = decay_ratio
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _stable(self, t_s: float) -> float:
        """Stable phase factor at t_s (time since the end of warmup), 1 at t_s = 0."""
        if self.stable_fn == 'constant':
            return 1.
        power = 0.5 if self.stable_fn == 'rsqrt' else self.stable_power
        return (1. + t_s / self.stable_timescale) ** -power

    def _decay(self, frac: float) -> float:
        """Decay phase factor at decay progress frac in [0, 1], from 1 to 0."""
        if callable(self.decay_fn):
            return self.decay_fn(frac)
        if self.decay_fn == 'linear':
            return 1. - frac
        if self.decay_fn == '1-sqrt':
            return 1. - math.sqrt(frac)
        if self.decay_fn == 'cosine':
            return 0.5 * (1. + math.cos(math.pi * frac))
        if self.decay_fn == 'power':
            return 1. - frac ** self.decay_power
        # exp: r ** frac decays to r at frac == 1, normalized to reach 0
        r = self.decay_ratio
        return (r ** frac - r) / (1. - r)

    def _get_lr(self, t: int) -> List[float]:
        if t < self.warmup_t:
            return [self.warmup_lr_init + t * s for s in self.warmup_steps]

        t_s = t - self.warmup_t
        # a decaying stable fn is kept at or above lr_min, the decay then has nothing left to do
        stable_lrs = [max(v * self._stable(t_s), self.lr_min) for v in self.base_values]
        t_decay = t_s - (self.t_initial - self.decay_t)
        if t_decay <= 0:
            return stable_lrs
        decay = self._decay(min(t_decay / self.decay_t, 1.))
        return [self.lr_min + (lr - self.lr_min) * decay for lr in stable_lrs]

    def get_cycle_length(self, cycles=0):
        return self.warmup_t + self.t_initial
