""" Scheduler Tests

Tests for learning rate schedulers in timm.scheduler.
"""
import math
import pytest
import torch
from torch.nn import Parameter

from timm.scheduler import (
    CosineLRScheduler,
    StepLRScheduler,
    MultiStepLRScheduler,
    PlateauLRScheduler,
    PolyLRScheduler,
    TanhLRScheduler,
    WSDLRScheduler,
    create_scheduler_v2,
)
from timm.scheduler.scheduler import Scheduler


def _create_optimizer(lr: float = 0.1, num_groups: int = 1) -> torch.optim.Optimizer:
    """Create a mock optimizer with simple parameters for testing."""
    param_groups = []
    for _ in range(num_groups):
        param = Parameter(torch.randn(10, 5))
        param_groups.append({'params': [param], 'lr': lr})
    optimizer = torch.optim.SGD(param_groups, lr=lr)
    return optimizer


class TestSchedulerBasics:
    """Test basic scheduler initialization and stepping."""

    @pytest.mark.parametrize('scheduler_cls,kwargs', [
        (CosineLRScheduler, {'t_initial': 100}),
        (StepLRScheduler, {'decay_t': 10}),
        (MultiStepLRScheduler, {'decay_t': [10, 20, 30]}),
        (PlateauLRScheduler, {}),
        (PolyLRScheduler, {'t_initial': 100}),
        (TanhLRScheduler, {'t_initial': 100}),
        (WSDLRScheduler, {'t_initial': 100, 'decay_t': 20}),
    ])
    def test_scheduler_init(self, scheduler_cls, kwargs):
        """Test that all schedulers can be initialized."""
        optimizer = _create_optimizer()
        scheduler = scheduler_cls(optimizer, **kwargs)
        assert scheduler is not None
        assert scheduler.optimizer is optimizer

    @pytest.mark.parametrize('scheduler_cls,kwargs', [
        (CosineLRScheduler, {'t_initial': 100}),
        (StepLRScheduler, {'decay_t': 10}),
        (MultiStepLRScheduler, {'decay_t': [10, 20, 30]}),
        (PolyLRScheduler, {'t_initial': 100}),
        (TanhLRScheduler, {'t_initial': 100}),
        (WSDLRScheduler, {'t_initial': 100, 'decay_t': 20}),
    ])
    def test_scheduler_step(self, scheduler_cls, kwargs):
        """Test that schedulers can step without error."""
        optimizer = _create_optimizer()
        scheduler = scheduler_cls(optimizer, **kwargs)

        initial_lr = optimizer.param_groups[0]['lr']
        for epoch in range(10):
            scheduler.step(epoch)

        # LR should have changed after stepping
        final_lr = optimizer.param_groups[0]['lr']
        # For most schedulers, LR should decrease or stay same
        assert final_lr <= initial_lr

    def test_plateau_scheduler_step(self):
        """Test PlateauLRScheduler with metric."""
        optimizer = _create_optimizer()
        scheduler = PlateauLRScheduler(optimizer, patience_t=2, decay_rate=0.5)

        # Simulate plateau - same metric for multiple steps
        for epoch in range(10):
            scheduler.step(epoch, metric=1.0)


class TestWarmup:
    """Test warmup behavior across schedulers."""

    @pytest.mark.parametrize('scheduler_cls,kwargs', [
        (CosineLRScheduler, {'t_initial': 100}),
        (StepLRScheduler, {'decay_t': 10}),
        (MultiStepLRScheduler, {'decay_t': [10, 20, 30]}),
        (PolyLRScheduler, {'t_initial': 100}),
        (TanhLRScheduler, {'t_initial': 100}),
        (WSDLRScheduler, {'t_initial': 100, 'decay_t': 20}),
    ])
    def test_warmup_lr_increases(self, scheduler_cls, kwargs):
        """Test that LR increases during warmup period."""
        base_lr = 0.1
        warmup_lr_init = 0.001
        warmup_t = 5

        optimizer = _create_optimizer(lr=base_lr)
        scheduler = scheduler_cls(
            optimizer,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
            **kwargs,
        )

        # Initial LR should be warmup_lr_init
        assert optimizer.param_groups[0]['lr'] == pytest.approx(warmup_lr_init, rel=1e-5)

        # LR should increase during warmup
        prev_lr = warmup_lr_init
        for epoch in range(1, warmup_t):
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            assert current_lr > prev_lr, f"LR should increase during warmup at epoch {epoch}"
            prev_lr = current_lr

    @pytest.mark.parametrize('scheduler_cls,kwargs', [
        (CosineLRScheduler, {'t_initial': 100}),
        (StepLRScheduler, {'decay_t': 10, 'decay_rate': 0.5}),
        (MultiStepLRScheduler, {'decay_t': [10, 20, 30], 'decay_rate': 0.5}),
        (PolyLRScheduler, {'t_initial': 100}),
        (TanhLRScheduler, {'t_initial': 100}),
    ])
    def test_warmup_prefix_reaches_target_lr(self, scheduler_cls, kwargs):
        """Test that target LR is reached at first step after warmup when warmup_prefix=True."""
        base_lr = 0.1
        warmup_lr_init = 0.001
        warmup_t = 5

        optimizer = _create_optimizer(lr=base_lr)
        scheduler = scheduler_cls(
            optimizer,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
            warmup_prefix=True,
            **kwargs,
        )

        # Step through warmup
        for epoch in range(warmup_t):
            scheduler.step(epoch)

        # At t=warmup_t (first step after warmup), with warmup_prefix=True,
        # the main schedule starts at t=0, which should be base_lr
        scheduler.step(warmup_t)
        lr_after_warmup = optimizer.param_groups[0]['lr']
        assert lr_after_warmup == pytest.approx(base_lr, rel=1e-5), \
            f"LR should be base_lr ({base_lr}) at first step after warmup, got {lr_after_warmup}"


class TestCosineScheduler:
    """Test CosineLRScheduler specific behavior."""

    def test_cosine_decay(self):
        """Test that cosine scheduler decays LR correctly."""
        base_lr = 0.1
        lr_min = 0.001
        t_initial = 100

        optimizer = _create_optimizer(lr=base_lr)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=lr_min,
        )

        # At t=0, LR should be base_lr
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr, rel=1e-5)

        # At t=t_initial/2, LR should be approximately (base_lr + lr_min) / 2
        scheduler.step(t_initial // 2)
        mid_lr = optimizer.param_groups[0]['lr']
        expected_mid = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * 0.5))
        assert mid_lr == pytest.approx(expected_mid, rel=1e-2)

        # At t=t_initial, LR should be lr_min
        scheduler.step(t_initial)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(lr_min, rel=1e-5)

    def test_cosine_cycles(self):
        """Test cosine scheduler with multiple cycles."""
        base_lr = 0.1
        lr_min = 0.001
        t_initial = 50
        cycle_limit = 2

        optimizer = _create_optimizer(lr=base_lr)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=lr_min,
            cycle_limit=cycle_limit,
        )

        # Step through the first cycle - at t_initial-1, LR should be near minimum
        scheduler.step(t_initial - 1)
        lr_near_end = optimizer.param_groups[0]['lr']
        assert lr_near_end < base_lr * 0.5, "LR should be significantly lower near end of cycle"

        # After cycle limit is exceeded, LR should stay at lr_min
        for epoch in range(t_initial * cycle_limit, t_initial * cycle_limit + 10):
            scheduler.step(epoch)
        lr_after_cycles = optimizer.param_groups[0]['lr']
        assert lr_after_cycles == pytest.approx(lr_min, rel=1e-5)

    def test_get_cycle_length(self):
        """Test get_cycle_length method."""
        optimizer = _create_optimizer()
        t_initial = 100

        scheduler = CosineLRScheduler(optimizer, t_initial=t_initial)
        assert scheduler.get_cycle_length(1) == t_initial

        # With warmup prefix
        warmup_t = 10
        scheduler_warmup = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            warmup_t=warmup_t,
            warmup_prefix=True,
        )
        assert scheduler_warmup.get_cycle_length(1) == t_initial + warmup_t


class TestStepScheduler:
    """Test StepLRScheduler specific behavior."""

    def test_step_decay(self):
        """Test that step scheduler decays at correct intervals."""
        base_lr = 0.1
        decay_t = 10
        decay_rate = 0.5

        optimizer = _create_optimizer(lr=base_lr)
        scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_t,
            decay_rate=decay_rate,
        )

        # Before first decay
        scheduler.step(decay_t - 1)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr, rel=1e-5)

        # After first decay
        scheduler.step(decay_t)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr * decay_rate, rel=1e-5)

        # After second decay
        scheduler.step(2 * decay_t)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr * decay_rate ** 2, rel=1e-5)


class TestMultiStepScheduler:
    """Test MultiStepLRScheduler specific behavior."""

    def test_multistep_decay(self):
        """Test decay at specified milestones."""
        base_lr = 0.1
        decay_t = [10, 20, 30]
        decay_rate = 0.5

        optimizer = _create_optimizer(lr=base_lr)
        scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_t,
            decay_rate=decay_rate,
        )

        # Before first milestone
        scheduler.step(8)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr, rel=1e-5)

        # After first milestone (step 10 means we've passed milestone at 10)
        scheduler.step(11)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr * decay_rate, rel=1e-5)

        # After second milestone
        scheduler.step(21)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr * decay_rate ** 2, rel=1e-5)

        # After third milestone
        scheduler.step(31)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr * decay_rate ** 3, rel=1e-5)


class TestPolyScheduler:
    """Test PolyLRScheduler specific behavior."""

    def test_poly_decay(self):
        """Test polynomial decay behavior."""
        base_lr = 0.1
        lr_min = 0.001
        t_initial = 100
        power = 1.0  # Linear decay

        optimizer = _create_optimizer(lr=base_lr)
        scheduler = PolyLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=lr_min,
            power=power,
        )

        # At t=0, LR should be base_lr
        assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr, rel=1e-5)

        # At t=t_initial, LR should be lr_min
        scheduler.step(t_initial)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(lr_min, rel=1e-5)


class TestTanhScheduler:
    """Test TanhLRScheduler specific behavior."""

    def test_tanh_decay(self):
        """Test tanh decay behavior."""
        base_lr = 0.1
        lr_min = 0.001
        t_initial = 100

        optimizer = _create_optimizer(lr=base_lr)
        scheduler = TanhLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=lr_min,
        )

        # Collect LR values
        lrs = [optimizer.param_groups[0]['lr']]
        for epoch in range(1, t_initial + 1):
            scheduler.step(epoch)
            lrs.append(optimizer.param_groups[0]['lr'])

        # LR should generally decrease (with possible non-monotonic behavior due to tanh)
        assert lrs[-1] < lrs[0]


class TestStateDict:
    """Test state dict save/load functionality."""

    @pytest.mark.parametrize('scheduler_cls,kwargs', [
        (CosineLRScheduler, {'t_initial': 100}),
        (StepLRScheduler, {'decay_t': 10}),
        (MultiStepLRScheduler, {'decay_t': [10, 20, 30]}),
        (PolyLRScheduler, {'t_initial': 100}),
        (TanhLRScheduler, {'t_initial': 100}),
    ])
    def test_state_dict_save_load(self, scheduler_cls, kwargs):
        """Test that state dict can be saved and loaded."""
        optimizer = _create_optimizer()
        scheduler = scheduler_cls(optimizer, **kwargs)

        # Step a few times
        for epoch in range(5):
            scheduler.step(epoch)

        # Save state
        state_dict = scheduler.state_dict()
        assert isinstance(state_dict, dict)

        # Create new scheduler and load state
        optimizer2 = _create_optimizer()
        scheduler2 = scheduler_cls(optimizer2, **kwargs)
        scheduler2.load_state_dict(state_dict)

        # State should be restored
        assert scheduler2.state_dict() == state_dict

    def test_plateau_state_dict_save_load(self):
        """Test PlateauLRScheduler state dict save/load."""
        optimizer = _create_optimizer()
        scheduler = PlateauLRScheduler(optimizer)

        # Step a few times
        for epoch in range(5):
            scheduler.step(epoch, metric=1.0)

        # Save state
        state_dict = scheduler.state_dict()
        assert isinstance(state_dict, dict)

        # Create new scheduler and load state
        optimizer2 = _create_optimizer()
        scheduler2 = PlateauLRScheduler(optimizer2)
        scheduler2.load_state_dict(state_dict)

        # State should be restored
        assert scheduler2.state_dict() == state_dict


class TestStepUpdate:
    """Test step_update for update-based scheduling."""

    @pytest.mark.parametrize('scheduler_cls,kwargs', [
        (CosineLRScheduler, {'t_initial': 100}),
        (StepLRScheduler, {'decay_t': 10, 'decay_rate': 0.5}),
        (MultiStepLRScheduler, {'decay_t': [10, 20, 30], 'decay_rate': 0.5}),
        (PolyLRScheduler, {'t_initial': 100}),
        (TanhLRScheduler, {'t_initial': 100}),
    ])
    def test_step_update_with_t_in_epochs_false(self, scheduler_cls, kwargs):
        """Test step_update when t_in_epochs=False."""
        optimizer = _create_optimizer()
        scheduler = scheduler_cls(
            optimizer,
            t_in_epochs=False,
            **kwargs,
        )

        initial_lr = optimizer.param_groups[0]['lr']

        # step_update should work when t_in_epochs=False
        for update in range(50):
            scheduler.step_update(update)

        # LR should have changed for all these schedulers by step 50
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr != initial_lr, f"LR should change after 50 updates for {scheduler_cls.__name__}"


class TestMultipleParamGroups:
    """Test schedulers with multiple parameter groups."""

    @pytest.mark.parametrize('scheduler_cls,kwargs', [
        (CosineLRScheduler, {'t_initial': 100}),
        (StepLRScheduler, {'decay_t': 10}),
        (MultiStepLRScheduler, {'decay_t': [10, 20, 30]}),
        (PolyLRScheduler, {'t_initial': 100}),
        (TanhLRScheduler, {'t_initial': 100}),
    ])
    def test_multiple_param_groups(self, scheduler_cls, kwargs):
        """Test that schedulers handle multiple param groups correctly."""
        optimizer = _create_optimizer(num_groups=3)
        scheduler = scheduler_cls(optimizer, **kwargs)

        initial_lrs = [pg['lr'] for pg in optimizer.param_groups]

        for epoch in range(20):
            scheduler.step(epoch)

        final_lrs = [pg['lr'] for pg in optimizer.param_groups]

        # All param groups should be updated
        for i, (initial, final) in enumerate(zip(initial_lrs, final_lrs)):
            assert final <= initial, f"Param group {i} LR should decrease or stay same"


class TestNoise:
    """Test noise application in schedulers."""

    @pytest.mark.parametrize('scheduler_cls,kwargs', [
        (CosineLRScheduler, {'t_initial': 100}),
        (StepLRScheduler, {'decay_t': 10}),
        (PolyLRScheduler, {'t_initial': 100}),
        (TanhLRScheduler, {'t_initial': 100}),
    ])
    def test_noise_range(self, scheduler_cls, kwargs):
        """Test that noise is applied within specified range."""
        optimizer = _create_optimizer()
        noise_range_t = (10, 50)

        scheduler = scheduler_cls(
            optimizer,
            noise_range_t=noise_range_t,
            noise_pct=0.5,
            noise_seed=42,
            **kwargs,
        )

        # Collect LRs with same seed - should be deterministic
        lrs_run1 = []
        for epoch in range(60):
            scheduler.step(epoch)
            lrs_run1.append(optimizer.param_groups[0]['lr'])

        # Reset and run again with same seed
        optimizer2 = _create_optimizer()
        scheduler2 = scheduler_cls(
            optimizer2,
            noise_range_t=noise_range_t,
            noise_pct=0.5,
            noise_seed=42,
            **kwargs,
        )

        lrs_run2 = []
        for epoch in range(60):
            scheduler2.step(epoch)
            lrs_run2.append(optimizer2.param_groups[0]['lr'])

        # With same seed, noise should be deterministic
        assert lrs_run1 == lrs_run2

class TestKDecay:
    """Test k-decay option in cosine and poly schedulers."""

    def test_cosine_k_decay(self):
        """Test k-decay in cosine scheduler."""
        optimizer1 = _create_optimizer()
        optimizer2 = _create_optimizer()

        scheduler_k1 = CosineLRScheduler(optimizer1, t_initial=100, k_decay=1.0)
        scheduler_k2 = CosineLRScheduler(optimizer2, t_initial=100, k_decay=2.0)

        # Different k values should produce different schedules
        lrs_k1 = []
        lrs_k2 = []
        for epoch in range(100):
            scheduler_k1.step(epoch)
            scheduler_k2.step(epoch)
            lrs_k1.append(optimizer1.param_groups[0]['lr'])
            lrs_k2.append(optimizer2.param_groups[0]['lr'])

        # The schedules should differ (except at endpoints)
        assert lrs_k1[50] != lrs_k2[50]

    def test_poly_k_decay(self):
        """Test k-decay in poly scheduler."""
        optimizer1 = _create_optimizer()
        optimizer2 = _create_optimizer()

        scheduler_k1 = PolyLRScheduler(optimizer1, t_initial=100, k_decay=1.0)
        scheduler_k2 = PolyLRScheduler(optimizer2, t_initial=100, k_decay=2.0)

        lrs_k1 = []
        lrs_k2 = []
        for epoch in range(100):
            scheduler_k1.step(epoch)
            scheduler_k2.step(epoch)
            lrs_k1.append(optimizer1.param_groups[0]['lr'])
            lrs_k2.append(optimizer2.param_groups[0]['lr'])

        # The schedules should differ
        assert lrs_k1[50] != lrs_k2[50]


class TestWSDScheduler:
    """Test WSDLRScheduler (warmup-stable-decay) behavior."""

    @staticmethod
    def _lrs(scheduler, optimizer, num_steps):
        lrs = []
        for t in range(num_steps):
            scheduler.step(t)
            lrs.append(optimizer.param_groups[0]['lr'])
        return lrs

    def test_phases(self):
        """Warmup to base lr, constant stable phase, decay reaching lr_min at warmup_t + t_initial, then held."""
        base_lr, lr_min, warmup_t, t_initial, decay_t = 0.1, 0.01, 5, 100, 20
        optimizer = _create_optimizer(lr=base_lr)
        scheduler = WSDLRScheduler(
            optimizer, t_initial=t_initial, decay_t=decay_t, lr_min=lr_min, warmup_t=warmup_t, warmup_lr_init=0.)
        assert scheduler.get_cycle_length() == warmup_t + t_initial
        lrs = self._lrs(scheduler, optimizer, warmup_t + t_initial + 10)
        assert lrs[warmup_t - 1] < base_lr
        # stable phase, from the end of warmup up to the start of the decay
        for t in range(warmup_t, warmup_t + t_initial - decay_t + 1):
            assert lrs[t] == pytest.approx(base_lr)
        # strictly decreasing decay, to lr_min at the end of the schedule, held after
        decay = lrs[warmup_t + t_initial - decay_t:warmup_t + t_initial + 1]
        assert all(a > b for a, b in zip(decay, decay[1:]))
        assert decay[-1] == pytest.approx(lr_min)
        assert all(lr == pytest.approx(lr_min) for lr in lrs[warmup_t + t_initial:])

    @pytest.mark.parametrize('decay_fn,mid_factor', [
        ('linear', 0.5),
        ('1-sqrt', 1 - math.sqrt(0.5)),
        ('cosine', 0.5),
        ('power', 1 - 0.5 ** 2),  # decay_power=2 below
        ('exp', (0.1 ** 0.5 - 0.1) / 0.9),  # decay_ratio=0.1
        (lambda frac: 1 - frac ** 3, 1 - 0.5 ** 3),
    ])
    def test_decay_fns(self, decay_fn, mid_factor):
        """Decay functions are 1 at the start, the expected value halfway and 0 at the end of the decay."""
        base_lr, t_initial, decay_t = 0.1, 100, 20
        optimizer = _create_optimizer(lr=base_lr)
        scheduler = WSDLRScheduler(optimizer, t_initial=t_initial, decay_t=decay_t, decay_fn=decay_fn, decay_power=2.)
        lrs = self._lrs(scheduler, optimizer, t_initial + 1)
        assert lrs[t_initial - decay_t] == pytest.approx(base_lr)
        assert lrs[t_initial - decay_t // 2] == pytest.approx(base_lr * mid_factor)
        assert lrs[t_initial] == pytest.approx(0., abs=1e-12)

    @pytest.mark.parametrize('stable_fn,timescale,power', [('rsqrt', 2, 0.5), ('rsqrt', 50, 0.5), ('power', 10, 1.)])
    def test_stable_fns(self, stable_fn, timescale, power):
        """Decaying stable fns hand off from warmup at the base lr for any timescale, and keep decaying through the
        decay phase (multiplicative w/ the decay fn, as per the Big Vision rsqrt + cooldown schedule)."""
        base_lr, warmup_t, t_initial, decay_t = 0.1, 5, 100, 20
        optimizer = _create_optimizer(lr=base_lr)
        scheduler = WSDLRScheduler(
            optimizer, t_initial=t_initial, decay_t=decay_t, stable_fn=stable_fn, stable_timescale=timescale,
            stable_power=power, decay_fn='linear', warmup_t=warmup_t,
        )
        lrs = self._lrs(scheduler, optimizer, warmup_t + t_initial + 1)
        assert lrs[warmup_t] == pytest.approx(base_lr)
        for t in range(warmup_t, warmup_t + t_initial - decay_t):
            t_s = t - warmup_t
            assert lrs[t] == pytest.approx(base_lr * (1 + t_s / timescale) ** -power)
        t = warmup_t + t_initial - decay_t // 2
        t_s = t - warmup_t
        assert lrs[t] == pytest.approx(base_lr * (1 + t_s / timescale) ** -power * 0.5)
        assert lrs[warmup_t + t_initial] == pytest.approx(0., abs=1e-12)

    def test_rsqrt_matches_big_vision(self):
        """With timescale == warmup_t the rsqrt stable fn is sqrt(timescale / max(t, timescale)) as in Big Vision."""
        base_lr, warmup_t, t_initial = 0.1, 10, 100
        optimizer = _create_optimizer(lr=base_lr)
        scheduler = WSDLRScheduler(optimizer, t_initial=t_initial, decay_t=1, stable_fn='rsqrt', warmup_t=warmup_t)
        lrs = self._lrs(scheduler, optimizer, warmup_t + t_initial - 1)
        for t in range(warmup_t, warmup_t + t_initial - 1):
            assert lrs[t] == pytest.approx(base_lr * math.sqrt(warmup_t / max(t, warmup_t)))

    def test_lr_min_floor(self):
        """A decaying stable fn does not go below lr_min."""
        optimizer = _create_optimizer(lr=0.1)
        scheduler = WSDLRScheduler(
            optimizer, t_initial=1000, decay_t=100, lr_min=0.05, stable_fn='power', stable_timescale=1, stable_power=2.)
        lrs = self._lrs(scheduler, optimizer, 1000)
        assert min(lrs) == pytest.approx(0.05)

    def test_invalid_args(self):
        optimizer = _create_optimizer()
        with pytest.raises(ValueError):
            WSDLRScheduler(optimizer, t_initial=100, decay_t=0)
        with pytest.raises(ValueError):
            WSDLRScheduler(optimizer, t_initial=100, decay_t=101)
        with pytest.raises(ValueError):
            WSDLRScheduler(optimizer, t_initial=100, decay_t=10, decay_fn='sqrt')
        with pytest.raises(ValueError):
            WSDLRScheduler(optimizer, t_initial=100, decay_t=10, stable_fn='rsqrt')  # no warmup, needs a timescale
        with pytest.raises(ValueError):
            WSDLRScheduler(optimizer, t_initial=100, decay_t=10, decay_fn='power', decay_power=0.)
        with pytest.raises(ValueError):
            WSDLRScheduler(
                optimizer, t_initial=100, decay_t=10, stable_fn='power', stable_timescale=10, stable_power=-1.)

    @pytest.mark.parametrize('step_on_updates', [False, True])
    def test_factory_rsqrt_timescale(self, step_on_updates):
        """The rsqrt timescale is given in epochs (converted to updates w/ step_on_updates), required w/o warmup."""
        updates_per_epoch = 10
        scale = updates_per_epoch if step_on_updates else 1
        kwargs = dict(
            sched='rsqrt', num_epochs=100, warmup_epochs=0, step_on_epochs=not step_on_updates,
            updates_per_epoch=updates_per_epoch if step_on_updates else 0,
        )
        with pytest.raises(ValueError):
            create_scheduler_v2(_create_optimizer(lr=0.1), **kwargs)
        optimizer = _create_optimizer(lr=0.1)
        scheduler, _ = create_scheduler_v2(optimizer, stable_timescale=4, **kwargs)
        assert scheduler.stable_timescale == 4 * scale
        step = scheduler.step_update if step_on_updates else scheduler.step
        step(12 * scale)  # t = 3 timescales, before the decay phase
        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1 * 4 ** -0.5)

    @pytest.mark.parametrize('sched,expected', [('wsd', 20), ('rsqrt', 20), ('step', 90)])
    def test_factory_default_decay_epochs(self, sched, expected):
        """decay_epochs=None resolves per scheduler, the final 20% for wsd / rsqrt, 90 epoch intervals for step."""
        optimizer = _create_optimizer(lr=0.1)
        scheduler, _ = create_scheduler_v2(optimizer, sched=sched, num_epochs=105, warmup_epochs=5)
        assert scheduler.decay_t == expected

    @pytest.mark.parametrize('warmup_prefix', [False, True])
    @pytest.mark.parametrize('step_on_updates', [False, True])
    def test_factory(self, warmup_prefix, step_on_updates):
        """The factory keeps --epochs semantics consistent w/ the other schedulers (the scheduler itself is always
        warmup prefixed), and resolves decay_epochs < 1 as a fraction of the stable + decay length."""
        epochs, warmup, decay, cooldown, updates_per_epoch = 100, 5, 20, 3, 10
        base_lr = 0.1
        for decay_epochs in (decay, decay / (epochs if warmup_prefix else epochs - warmup)):
            optimizer = _create_optimizer(lr=base_lr)
            scheduler, num_epochs = create_scheduler_v2(
                optimizer, sched='wsd', num_epochs=epochs, decay_epochs=decay_epochs, warmup_epochs=warmup,
                warmup_prefix=warmup_prefix, cooldown_epochs=cooldown, step_on_epochs=not step_on_updates,
                updates_per_epoch=updates_per_epoch if step_on_updates else 0,
            )
            assert num_epochs == epochs + cooldown + (warmup if warmup_prefix else 0)
            scale = updates_per_epoch if step_on_updates else 1
            assert scheduler.warmup_t == warmup * scale
            assert scheduler.decay_t == decay * scale
            assert scheduler.get_cycle_length() == (num_epochs - cooldown) * scale
            # last scheduled step (before the cooldown hold) is just above lr_min, the cooldown is at lr_min
            step = scheduler.step_update if step_on_updates else scheduler.step
            step((num_epochs - cooldown) * scale - 1)
            assert 0 < optimizer.param_groups[0]['lr'] < base_lr * 0.5
            step((num_epochs - cooldown) * scale)
            assert optimizer.param_groups[0]['lr'] == pytest.approx(0., abs=1e-12)

    def test_factory_rsqrt(self):
        optimizer = _create_optimizer(lr=0.1)
        scheduler, _ = create_scheduler_v2(optimizer, sched='rsqrt', num_epochs=100, decay_epochs=10, warmup_epochs=5)
        assert scheduler.stable_fn == 'rsqrt' and scheduler.stable_timescale == 5 and scheduler.decay_fn == 'linear'
        with pytest.raises(ValueError):
            create_scheduler_v2(optimizer, sched='wsd', num_epochs=10, decay_epochs=10, warmup_epochs=5)
