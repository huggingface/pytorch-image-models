"""Scheduled batch sampling and transform dispatch for map-style datasets."""

import math
from itertools import islice
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, Sampler


class ScheduledBatchSampler(Sampler[List[Tuple[Any, int]]]):
    """Batch sampler that associates each batch with a scheduled transform choice.

    In sample-budget mode, a canonical composition of ``(choice index, batch
    size)`` pairs is created once and shuffled at the start of each iteration.
    Fixed-batch progressive mode creates a schedule for the current epoch,
    moving a probability window from the first choice to the last while
    retaining configurable random exploration.

    The same seed, sampler length, and epoch produce an identical batch-shape
    schedule on every distributed rank. If a fixed number of batches requires
    more indices than the wrapped sampler provides, its index stream is cycled.

    The yielded indices are ``(sample index, choice index)`` pairs intended for
    use with :class:`ScheduledTransformDataset`.

    Args:
        sampler: Base sample-index sampler.
        batch_sizes: Batch size corresponding to each transform choice.
        choice_weights: Base sampling weights for the choices. A zero weight
            excludes that choice from all schedules.
        seed: Seed shared by schedule creation and shuffling.
        drop_last: Drop an undersized tail in sample-budget mode.
        shuffle_schedule: Shuffle batch order within each epoch.
        num_batches: Fixed number of batches per epoch. Progressive mode infers
            this from its training-average expected batch size when omitted.
        choice_schedule: Either ``constant`` or ``progressive``.
        schedule_epochs: Number of epochs over which progressive choices move
            from the first choice to the last.
        schedule_spread: Progressive probability-window standard deviation in
            choice-index units. Zero selects only the nearest choice.
        schedule_random_mix: Fraction of uniform exploration over active
            choices mixed into the progressive choice probabilities.
    """

    def __init__(
            self,
            sampler: Sampler,
            batch_sizes: Sequence[int],
            choice_weights: Optional[Sequence[float]] = None,
            seed: int = 0,
            drop_last: bool = True,
            shuffle_schedule: bool = True,
            num_batches: Optional[int] = None,
            choice_schedule: str = 'constant',
            schedule_epochs: Optional[int] = None,
            schedule_spread: float = 0.65,
            schedule_random_mix: float = 0.1,
    ) -> None:
        if not hasattr(sampler, '__len__'):
            raise TypeError('ScheduledBatchSampler requires a sampler with a length.')
        if len(sampler) <= 0:
            raise ValueError('ScheduledBatchSampler requires a non-empty sampler.')
        if not batch_sizes:
            raise ValueError('batch_sizes must contain at least one value.')
        if any(int(batch_size) != batch_size or batch_size <= 0 for batch_size in batch_sizes):
            raise ValueError('All scheduled batch sizes must be positive integers.')
        if num_batches is not None and (int(num_batches) != num_batches or num_batches <= 0):
            raise ValueError('num_batches must be a positive integer when specified.')
        if choice_schedule not in ('constant', 'progressive'):
            raise ValueError("choice_schedule must be 'constant' or 'progressive'.")
        if schedule_spread < 0:
            raise ValueError('schedule_spread must be non-negative.')
        if not 0 <= schedule_random_mix <= 1:
            raise ValueError('schedule_random_mix must be between 0 and 1.')
        if choice_schedule == 'progressive':
            if schedule_epochs is None or int(schedule_epochs) != schedule_epochs or schedule_epochs <= 0:
                raise ValueError('schedule_epochs must be a positive integer for a progressive schedule.')

        self.sampler = sampler
        self.batch_sizes = tuple(int(batch_size) for batch_size in batch_sizes)
        self.choice_weights = self._normalize_choice_weights(choice_weights)
        self._active_choices = tuple(
            choice_index
            for choice_index, choice_weight in enumerate(self.choice_weights)
            if choice_weight > 0
        )
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle_schedule = shuffle_schedule
        self.choice_schedule = choice_schedule
        self.schedule_epochs = int(schedule_epochs) if schedule_epochs is not None else None
        self.schedule_spread = schedule_spread
        self.schedule_random_mix = schedule_random_mix
        self.epoch = 0
        self.average_batch_size = self._calculate_average_batch_size()
        if choice_schedule == 'progressive' and num_batches is None:
            num_batches = self._infer_num_batches()
        self.num_batches = int(num_batches) if num_batches is not None else None
        self._sample_budget_schedule: Tuple[Tuple[int, int], ...] = ()
        if self.num_batches is None:
            self._sample_budget_schedule = self._create_sample_budget_schedule()
            if not self._sample_budget_schedule:
                raise ValueError(
                    'No full scheduled batch fits the sampler; reduce the batch sizes.'
                )

    def _normalize_choice_weights(
            self,
            choice_weights: Optional[Sequence[float]],
    ) -> torch.Tensor:
        if choice_weights is None:
            return torch.full((len(self.batch_sizes),), 1.0 / len(self.batch_sizes), dtype=torch.float64)
        if len(choice_weights) != len(self.batch_sizes):
            raise ValueError('choice_weights and batch_sizes must have the same length.')

        weights = torch.tensor(choice_weights, dtype=torch.float64)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('choice_weights must contain finite, non-negative values.')
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError('choice_weights must have a positive sum.')
        return weights / weight_sum

    def choice_weights_for_epoch(self, epoch: int) -> torch.Tensor:
        """Return normalized choice weights for an epoch.

        Args:
            epoch: Zero-based training epoch.

        Returns:
            Normalized floating-point choice weights.
        """
        if self.choice_schedule == 'constant' or len(self.batch_sizes) == 1:
            return self.choice_weights

        if self.schedule_epochs == 1:
            progress = 1.0
        else:
            progress = min(max(epoch / (self.schedule_epochs - 1), 0.0), 1.0)
        choice_positions = torch.arange(len(self.batch_sizes), dtype=torch.float64)
        center = progress * (len(self.batch_sizes) - 1)

        if self.schedule_spread == 0:
            distances = (choice_positions - center).abs()
            curriculum_weights = (distances == distances.min()).to(torch.float64)
        else:
            curriculum_weights = torch.exp(
                -0.5 * ((choice_positions - center) / self.schedule_spread) ** 2
            )
        curriculum_weights *= self.choice_weights
        if curriculum_weights.sum() <= 0:
            nearest_active = min(self._active_choices, key=lambda index: abs(index - center))
            curriculum_weights = torch.zeros_like(self.choice_weights)
            curriculum_weights[nearest_active] = 1.0
        else:
            curriculum_weights /= curriculum_weights.sum()

        if self.schedule_random_mix:
            random_weights = (self.choice_weights > 0).to(curriculum_weights.dtype)
            random_weights /= random_weights.sum()
            curriculum_weights = (
                1.0 - self.schedule_random_mix
            ) * curriculum_weights + self.schedule_random_mix * random_weights
        return curriculum_weights / curriculum_weights.sum()

    def _calculate_average_batch_size(self) -> float:
        batch_sizes = torch.tensor(self.batch_sizes, dtype=torch.float64)
        if self.choice_schedule == 'progressive':
            expected_batch_sizes = [
                torch.dot(self.choice_weights_for_epoch(epoch), batch_sizes) for epoch in range(self.schedule_epochs)
            ]
            return float(torch.stack(expected_batch_sizes).mean().item())
        return float(torch.dot(self.choice_weights, batch_sizes).item())

    def _infer_num_batches(self) -> int:
        if len(set(self.batch_sizes)) == 1:
            batch_size = self.batch_sizes[0]
            if self.drop_last:
                num_batches = len(self.sampler) // batch_size
            else:
                num_batches = math.ceil(len(self.sampler) / batch_size)
        else:
            if self.drop_last:
                num_batches = int(len(self.sampler) / self.average_batch_size)
            else:
                num_batches = math.ceil(len(self.sampler) / self.average_batch_size)
        if num_batches < 1:
            raise ValueError('No full scheduled batch fits the sampler; reduce the batch sizes.')
        return num_batches

    def _sample_choice(
            self,
            generator: torch.Generator,
            valid_choices: Optional[Sequence[int]] = None,
            choice_weights: Optional[torch.Tensor] = None,
    ) -> int:
        choice_weights = self.choice_weights if choice_weights is None else choice_weights
        if valid_choices is None:
            return int(torch.multinomial(choice_weights, 1, generator=generator).item())

        valid_choices = tuple(valid_choices)
        weights = choice_weights[list(valid_choices)]
        if weights.sum() <= 0:
            raise RuntimeError('No positive-weight scheduled choice is available for this batch.')
        sampled_index = int(torch.multinomial(weights, 1, generator=generator).item())
        return valid_choices[sampled_index]

    def _create_sample_budget_schedule(self) -> Tuple[Tuple[int, int], ...]:
        generator = torch.Generator().manual_seed(self.seed)
        remaining = len(self.sampler)
        min_batch_size = min(self.batch_sizes[index] for index in self._active_choices)
        schedule = []

        while remaining >= min_batch_size:
            valid_choices = [
                choice_index for choice_index in self._active_choices if self.batch_sizes[choice_index] <= remaining
            ]
            choice_index = self._sample_choice(generator, valid_choices)
            batch_size = self.batch_sizes[choice_index]
            schedule.append((choice_index, batch_size))
            remaining -= batch_size

        if remaining and not self.drop_last:
            schedule.append((self._sample_choice(generator), remaining))

        return tuple(schedule)

    def _create_fixed_batch_schedule(self, epoch: int) -> Tuple[Tuple[int, int], ...]:
        generator = torch.Generator().manual_seed(self.seed + 2 * epoch)
        choice_weights = self.choice_weights_for_epoch(epoch)
        schedule = []
        for _ in range(self.num_batches):
            choice_index = self._sample_choice(generator, choice_weights=choice_weights)
            schedule.append((choice_index, self.batch_sizes[choice_index]))
        return tuple(schedule)

    def _create_schedule(self, epoch: int) -> Tuple[Tuple[int, int], ...]:
        if self.num_batches is not None:
            return self._create_fixed_batch_schedule(epoch)
        return self._sample_budget_schedule

    @property
    def schedule(self) -> Tuple[Tuple[int, int], ...]:
        """Return the unshuffled schedule for the currently selected epoch."""
        return self._create_schedule(self.epoch)

    def _cycling_sampler(self) -> Iterator[Any]:
        while True:
            yielded = False
            for sample_index in self.sampler:
                yielded = True
                yield sample_index
            if not yielded:
                break

    def __iter__(self) -> Iterator[List[Tuple[Any, int]]]:
        epoch = self.epoch
        schedule = self._create_schedule(epoch)
        if self.shuffle_schedule and len(schedule) > 1:
            generator = torch.Generator().manual_seed(self.seed + 2 * epoch + 1)
            order = torch.randperm(len(schedule), generator=generator).tolist()
            schedule = tuple(schedule[index] for index in order)

        sample_indices = self._cycling_sampler() if self.num_batches is not None else iter(self.sampler)
        for choice_index, batch_size in schedule:
            batch = list(islice(sample_indices, batch_size))
            if len(batch) != batch_size:
                if self.drop_last or not batch:
                    break
            yield [(sample_index, choice_index) for sample_index in batch]

    def __len__(self) -> int:
        if self.num_batches is not None:
            return self.num_batches
        return len(self._sample_budget_schedule)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the next iteration and the wrapped sampler."""
        self.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)


class ScheduledTransformDataset(Dataset):
    """Apply one of several transforms selected by an annotated sample index.

    Args:
        dataset: Map-style base dataset returning tuple or list samples.
        transforms: Transforms indexed by the choice annotation supplied by
            :class:`ScheduledBatchSampler`.
    """

    def __init__(
            self,
            dataset: Dataset,
            transforms: Sequence[Callable],
    ) -> None:
        if not transforms:
            raise ValueError('transforms must contain at least one transform.')
        self.dataset = dataset
        self.transforms = tuple(transforms)

    def __getitem__(self, scheduled_index: Tuple[Any, int]) -> Union[Tuple[Any, ...], List[Any]]:
        sample_index, transform_index = scheduled_index
        if not 0 <= transform_index < len(self.transforms):
            raise IndexError(f'Transform index {transform_index} is out of range.')

        sample = self.dataset[sample_index]
        if not isinstance(sample, (tuple, list)) or not sample:
            raise TypeError('ScheduledTransformDataset expects tuple/list dataset samples.')
        image = self.transforms[transform_index](sample[0])
        if isinstance(sample, tuple):
            return image, *sample[1:]
        return [image, *sample[1:]]

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch updates to an epoch-aware base dataset."""
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

    def filename(self, index: int, basename: bool = False, absolute: bool = False) -> Any:
        return self.dataset.filename(index, basename=basename, absolute=absolute)

    def filenames(self, basename: bool = False, absolute: bool = False) -> Any:
        return self.dataset.filenames(basename=basename, absolute=absolute)
