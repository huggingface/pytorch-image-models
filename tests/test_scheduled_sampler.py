from collections import Counter

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, DistributedSampler, SequentialSampler

from timm.data import FastCollateMixup, ScheduledBatchSampler, ScheduledTransformDataset, create_loader


class _ImageDataset(Dataset):
    def __init__(self, length=64, image_size=48):
        self.length = length
        self.image_size = image_size
        self.transform = None

    def __getitem__(self, index):
        image = Image.new('RGB', (self.image_size, self.image_size), color=index % 256)
        if self.transform is not None:
            image = self.transform(image)
        return image, index

    def __len__(self):
        return self.length


def _batch_signature(batches):
    return [(batch[0][1], len(batch)) for batch in batches]


def test_scheduled_batch_sampler_stable_epoch_length_and_composition():
    sampler = SequentialSampler(range(101))
    batch_sampler = ScheduledBatchSampler(
        sampler,
        batch_sizes=(16, 8, 4),
        choice_weights=(0.2, 0.3, 0.5),
        seed=17,
    )

    epoch_0 = list(batch_sampler)
    batch_sampler.set_epoch(1)
    epoch_1 = list(batch_sampler)

    assert len(epoch_0) == len(epoch_1) == len(batch_sampler)
    assert Counter(_batch_signature(epoch_0)) == Counter(_batch_signature(epoch_1))
    assert _batch_signature(epoch_0) != _batch_signature(epoch_1)
    assert sum(map(len, epoch_0)) == sum(batch_size for _, batch_size in batch_sampler.schedule)
    assert 0 <= len(sampler) - sum(map(len, epoch_0)) < min(batch_sampler.batch_sizes)
    assert [sample_index for batch in epoch_0 for sample_index, _ in batch] == list(
        range(sum(map(len, epoch_0)))
    )
    assert all(len({choice for _, choice in batch}) == 1 for batch in epoch_0)


def test_sample_budget_schedule_is_created_once_and_cached():
    class TrackingScheduledBatchSampler(ScheduledBatchSampler):
        def __init__(self, *args, **kwargs):
            self.schedule_create_count = 0
            super().__init__(*args, **kwargs)

        def _create_sample_budget_schedule(self):
            self.schedule_create_count += 1
            return super()._create_sample_budget_schedule()

    batch_sampler = TrackingScheduledBatchSampler(
        SequentialSampler(range(101)),
        batch_sizes=(16, 8, 4),
        seed=17,
    )

    assert batch_sampler.schedule_create_count == 1
    assert len(batch_sampler) == len(batch_sampler) == len(list(batch_sampler))
    assert batch_sampler.schedule_create_count == 1


def test_constant_schedule_ignores_progressive_only_options():
    batch_sampler = ScheduledBatchSampler(
        SequentialSampler(range(32)),
        batch_sizes=(8, 4),
        choice_schedule='constant',
        schedule_epochs=0,
        schedule_spread=-1,
        schedule_random_mix=2,
    )

    assert len(batch_sampler) > 0


def test_zero_weight_choices_are_excluded_from_progressive_random_mix_and_sample_budget():
    progressive_sampler = ScheduledBatchSampler(
        SequentialSampler(range(64)),
        batch_sizes=(16, 8, 4),
        choice_weights=(1, 0, 0),
        choice_schedule='progressive',
        schedule_epochs=3,
        schedule_random_mix=0.1,
    )

    for epoch in range(3):
        assert progressive_sampler.choice_weights_for_epoch(epoch)[1:].tolist() == [0, 0]

    snap_sampler = ScheduledBatchSampler(
        SequentialSampler(range(64)),
        batch_sizes=(16, 12, 8, 4),
        choice_weights=(1, 0, 0, 1),
        choice_schedule='progressive',
        schedule_epochs=4,
        schedule_spread=0,
        schedule_random_mix=0,
    )
    assert snap_sampler.choice_weights_for_epoch(1).tolist() == [1, 0, 0, 0]
    assert snap_sampler.choice_weights_for_epoch(2).tolist() == [0, 0, 0, 1]

    sample_budget_sampler = ScheduledBatchSampler(
        SequentialSampler(range(12)),
        batch_sizes=(8, 4),
        choice_weights=(1, 0),
    )
    batches = list(sample_budget_sampler)
    assert _batch_signature(batches) == [(0, 8)]


def test_progressive_schedule_rejects_when_no_full_batch_fits():
    with pytest.raises(ValueError, match='No full scheduled batch'):
        ScheduledBatchSampler(
            SequentialSampler(range(3)),
            batch_sizes=(8, 4),
            choice_schedule='progressive',
            schedule_epochs=3,
        )


def test_sample_budget_schedule_rejects_when_no_full_batch_fits():
    with pytest.raises(ValueError, match='No full scheduled batch'):
        ScheduledBatchSampler(SequentialSampler(range(3)), batch_sizes=(8, 4))


def test_scheduled_batch_sampler_rejects_empty_sampler():
    with pytest.raises(ValueError, match='non-empty sampler'):
        ScheduledBatchSampler(SequentialSampler(range(0)), batch_sizes=(8, 4))


def test_scheduled_batch_sampler_distributed_shapes_match():
    dataset = list(range(96))
    rank_0_sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=11)
    rank_1_sampler = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, seed=11)
    rank_0_batches = ScheduledBatchSampler(rank_0_sampler, (12, 6), seed=29)
    rank_1_batches = ScheduledBatchSampler(rank_1_sampler, (12, 6), seed=29)

    rank_0_batches.set_epoch(3)
    rank_1_batches.set_epoch(3)
    rank_0_epoch = list(rank_0_batches)
    rank_1_epoch = list(rank_1_batches)

    assert _batch_signature(rank_0_epoch) == _batch_signature(rank_1_epoch)
    rank_0_indices = {index for batch in rank_0_epoch for index, _ in batch}
    rank_1_indices = {index for batch in rank_1_epoch for index, _ in batch}
    assert rank_0_indices.isdisjoint(rank_1_indices)


def test_progressive_schedule_moves_choices_and_preserves_constant_batch_budget():
    sampler = SequentialSampler(range(1000))
    batch_sampler = ScheduledBatchSampler(
        sampler,
        batch_sizes=(10, 10, 10),
        seed=23,
        choice_schedule='progressive',
        schedule_epochs=5,
        schedule_spread=0.5,
        schedule_random_mix=0.1,
    )

    assert batch_sampler.average_batch_size == 10
    assert len(batch_sampler) == 100
    start_weights = batch_sampler.choice_weights_for_epoch(0)
    middle_weights = batch_sampler.choice_weights_for_epoch(2)
    end_weights = batch_sampler.choice_weights_for_epoch(4)
    assert start_weights.argmax().item() == 0
    assert middle_weights.argmax().item() == 1
    assert end_weights.argmax().item() == 2
    assert (start_weights >= 0.1 / 3).all()

    epoch_0 = list(batch_sampler)
    batch_sampler.set_epoch(4)
    epoch_4 = list(batch_sampler)
    assert len(epoch_0) == len(epoch_4) == 100
    assert sum(map(len, epoch_0)) == sum(map(len, epoch_4)) == 1000
    assert [index for batch in epoch_0 for index, _ in batch] == list(range(1000))
    assert [index for batch in epoch_4 for index, _ in batch] == list(range(1000))
    assert sum(choice for choice, _ in _batch_signature(epoch_0)) < sum(
        choice
        for choice, _ in _batch_signature(epoch_4)
    )


def test_progressive_schedule_is_created_when_iteration_starts():
    class TrackingScheduledBatchSampler(ScheduledBatchSampler):
        def __init__(self, *args, **kwargs):
            self.created_epochs = []
            super().__init__(*args, **kwargs)

        def _create_fixed_batch_schedule(self, epoch):
            self.created_epochs.append(epoch)
            return super()._create_fixed_batch_schedule(epoch)

    batch_sampler = TrackingScheduledBatchSampler(
        SequentialSampler(range(32)),
        batch_sizes=(8, 4),
        num_batches=4,
        choice_schedule='progressive',
        schedule_epochs=3,
    )

    assert batch_sampler.created_epochs == []
    batch_sampler.set_epoch(2)
    assert batch_sampler.created_epochs == []

    iterator = iter(batch_sampler)
    assert batch_sampler.created_epochs == []
    next(iterator)
    assert batch_sampler.created_epochs == [2]


def test_progressive_schedule_infers_policy_average_batch_budget_and_cycles_indices():
    sampler = SequentialSampler(range(1000))
    batch_sampler = ScheduledBatchSampler(
        sampler,
        batch_sizes=(80, 40, 20),
        seed=3,
        choice_schedule='progressive',
        schedule_epochs=5,
        schedule_spread=0.5,
        schedule_random_mix=0.1,
    )
    batch_sizes = torch.tensor(batch_sampler.batch_sizes, dtype=torch.float64)
    expected_average = (
        torch
        .stack([torch.dot(batch_sampler.choice_weights_for_epoch(epoch), batch_sizes) for epoch in range(5)])
        .mean()
        .item()
    )

    assert batch_sampler.average_batch_size == pytest.approx(expected_average)
    assert len(batch_sampler) == int(len(sampler) / expected_average)

    epoch_0 = list(batch_sampler)
    epoch_0_indices = [index for batch in epoch_0 for index, _ in batch]
    assert len(epoch_0_indices) > len(sampler)
    assert epoch_0_indices[: len(sampler)] == list(range(len(sampler)))
    assert epoch_0_indices[len(sampler) :] == list(range(len(epoch_0_indices) - len(sampler)))

    batch_sampler.set_epoch(4)
    epoch_4 = list(batch_sampler)
    assert len(epoch_4) == len(epoch_0)
    assert sum(map(len, epoch_4)) < len(sampler)


def test_progressive_schedule_distributed_shapes_match_and_advance():
    dataset = list(range(96))
    samplers = [DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=11) for rank in range(2)]
    batch_samplers = [
        ScheduledBatchSampler(
            sampler,
            batch_sizes=(12, 6, 3),
            seed=29,
            num_batches=8,
            choice_schedule='progressive',
            schedule_epochs=3,
            schedule_spread=0,
            schedule_random_mix=0,
        )
        for sampler in samplers
    ]

    epoch_0 = [list(batch_sampler) for batch_sampler in batch_samplers]
    assert _batch_signature(epoch_0[0]) == _batch_signature(epoch_0[1])
    assert {choice for choice, _ in _batch_signature(epoch_0[0])} == {0}
    for batch_sampler in batch_samplers:
        batch_sampler.set_epoch(2)
    epoch_2 = [list(batch_sampler) for batch_sampler in batch_samplers]
    assert _batch_signature(epoch_2[0]) == _batch_signature(epoch_2[1])
    assert {choice for choice, _ in _batch_signature(epoch_2[0])} == {2}

    for rank_batches in (*epoch_0, *epoch_2):
        assert len(rank_batches) == 8
    for rank_epochs in (epoch_0, epoch_2):
        rank_indices = [{index for batch in rank_batches for index, _ in batch} for rank_batches in rank_epochs]
        assert rank_indices[0].isdisjoint(rank_indices[1])


def test_scheduled_transform_dataset_preserves_sample_fields():
    class ExtraFieldDataset(Dataset):
        def __getitem__(self, index):
            return index, index + 1, f'sample-{index}'

        def __len__(self):
            return 4

    dataset = ScheduledTransformDataset(ExtraFieldDataset(), (lambda value: value * 2, lambda value: value * 3))

    assert dataset[(2, 0)] == (4, 3, 'sample-2')
    assert dataset[(2, 1)] == (6, 3, 'sample-2')


def test_create_loader_scheduled_resolutions_and_batch_sizes():
    loader = create_loader(
        _ImageDataset(),
        input_size=(3, 32, 32),
        batch_size=4,
        input_size_choices=(16, 32),
        batch_size_choices=(8, 4),
        batch_choice_weights=(0.5, 0.5),
        batch_choice_seed=7,
        is_training=True,
        no_aug=True,
        use_prefetcher=False,
        num_workers=0,
        persistent_workers=False,
    )

    assert isinstance(loader.dataset, ScheduledTransformDataset)
    assert isinstance(loader.batch_sampler, ScheduledBatchSampler)
    for images, targets in loader:
        expected_batch_size = {16: 8, 32: 4}[images.shape[-1]]
        assert images.shape == (expected_batch_size, 3, images.shape[-2], images.shape[-1])
        assert images.shape[-2] == images.shape[-1]
        assert targets.shape == (expected_batch_size,)


def test_create_loader_scheduled_resolutions_default_to_batch_size():
    loader = create_loader(
        _ImageDataset(),
        input_size=(3, 32, 32),
        batch_size=4,
        input_size_choices=(16, 32),
        is_training=True,
        no_aug=True,
        use_prefetcher=False,
        num_workers=0,
        persistent_workers=False,
    )

    assert all(images.shape[0] == 4 for images, _ in loader)


def test_create_loader_progressive_resolutions_keep_default_batch_and_sample_counts():
    loader = create_loader(
        _ImageDataset(length=96),
        input_size=(3, 32, 32),
        batch_size=4,
        input_size_choices=(16, 24, 32),
        batch_choice_schedule='progressive',
        batch_schedule_epochs=3,
        batch_schedule_spread=0,
        batch_schedule_random_mix=0,
        is_training=True,
        no_aug=True,
        use_prefetcher=False,
        num_workers=0,
        persistent_workers=False,
    )

    assert len(loader) == 24
    for epoch, expected_size in enumerate((16, 24, 32)):
        loader.batch_sampler.set_epoch(epoch)
        epoch_batches = list(loader)
        assert len(epoch_batches) == 24
        assert sum(images.shape[0] for images, _ in epoch_batches) == 96
        assert {images.shape[-1] for images, _ in epoch_batches} == {expected_size}
        assert {images.shape[0] for images, _ in epoch_batches} == {4}


def test_create_loader_scheduled_resolutions_with_prefetch_mixup():
    loader = create_loader(
        _ImageDataset(),
        input_size=(3, 32, 32),
        batch_size=4,
        input_size_choices=(16, 32),
        batch_size_choices=(8, 4),
        is_training=True,
        no_aug=True,
        collate_fn=FastCollateMixup(mixup_alpha=0.2, num_classes=64),
        use_prefetcher=True,
        device=torch.device('cpu'),
        num_workers=0,
        persistent_workers=False,
    )

    assert isinstance(loader.batch_sampler, ScheduledBatchSampler)
    for images, targets in loader:
        assert images.dtype == torch.float32
        assert images.shape[0] == targets.shape[0]
        assert targets.shape[1] == 64


def test_create_loader_rejects_scheduled_resolutions_with_multi_epochs_loader():
    with pytest.raises(ValueError, match='MultiEpochsDataLoader'):
        create_loader(
            _ImageDataset(length=32),
            input_size=(3, 32, 32),
            batch_size=4,
            input_size_choices=(16, 32),
            is_training=True,
            no_aug=True,
            use_prefetcher=False,
            use_multi_epochs_loader=True,
            num_workers=0,
            persistent_workers=False,
        )


@pytest.mark.parametrize(
    'scheduled_option',
    (
        {'batch_choice_seed': 7},
        {'batch_schedule_epochs': 3},
        {'batch_schedule_spread': 0.5},
        {'batch_schedule_random_mix': 0.2},
    ),
)
def test_create_loader_ignores_scheduled_options_without_input_size_choices(scheduled_option):
    loader = create_loader(
        _ImageDataset(length=32),
        input_size=(3, 32, 32),
        batch_size=4,
        is_training=True,
        no_aug=True,
        use_prefetcher=False,
        num_workers=0,
        persistent_workers=False,
        **scheduled_option,
    )

    assert loader.batch_size == 4


def test_create_loader_standard_batching_path_is_unchanged():
    dataset = _ImageDataset()
    loader = create_loader(
        dataset,
        input_size=(3, 32, 32),
        batch_size=4,
        is_training=True,
        no_aug=True,
        use_prefetcher=False,
        num_workers=0,
        persistent_workers=False,
    )

    images, targets = next(iter(loader))
    assert loader.dataset is dataset
    assert loader.batch_size == 4
    assert images.shape == (4, 3, 32, 32)
    assert targets.shape == (4,)
