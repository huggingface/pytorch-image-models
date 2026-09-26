import pytest

from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler


@pytest.mark.parametrize('size, replicas', [(1, 4), (3, 8)])
def test_ordered_sampler_pads_tiny_dataset(size, replicas):
    dataset = list(range(size))
    samplers = [OrderedDistributedSampler(dataset, num_replicas=replicas, rank=rank)
                for rank in range(replicas)]
    samples = [list(sampler) for sampler in samplers]

    assert all(len(indices) == 1 for indices in samples)
    assert [indices[0] for indices in samples] == [i % size for i in range(replicas)]
    assert [sampler.num_valid_samples for sampler in samplers] == [int(rank < size) for rank in range(replicas)]


@pytest.mark.parametrize('size, replicas, expected', [(64, 2, 32), (1, 8, 1), (256, 2, 128), (257, 2, 128)])
def test_repeat_aug_sampler_uses_tiny_dataset(size, replicas, expected):
    dataset = list(range(size))
    samples = [list(RepeatAugSampler(dataset, num_replicas=replicas, rank=rank))
               for rank in range(replicas)]

    assert all(len(indices) == expected for indices in samples)
    assert all(0 <= index < size for indices in samples for index in indices)
