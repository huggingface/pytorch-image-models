import torch
from torch.utils.data import DataLoader, Dataset

from timm.data import NaFlexMapDatasetWrapper


class _TensorImageDataset(Dataset):

    def __init__(self, length=32):
        self.length = length

    def __getitem__(self, index):
        return torch.full((3, 2, 2), index, dtype=torch.float32), index

    def __len__(self):
        return self.length


def _create_naflex_dataset(epoch=0):
    return NaFlexMapDatasetWrapper(
        _TensorImageDataset(),
        patch_size=1,
        seq_lens=(1,),
        max_tokens_per_batch=4,
        seed=17,
        epoch=epoch,
        batch_divisor=1,
    )


def _loader_indices(loader):
    return [index for _, targets in loader for index in targets.tolist()]


def test_naflex_epoch_batches_are_prepared_when_iteration_starts():
    dataset = _create_naflex_dataset()
    prepare_epoch_batches = dataset._prepare_epoch_batches
    prepared_epochs = []

    def prepare(epoch):
        prepared_epochs.append(epoch)
        return prepare_epoch_batches(epoch)

    dataset._prepare_epoch_batches = prepare

    assert prepared_epochs == []
    dataset.set_epoch(3)
    assert prepared_epochs == []
    assert dataset.shared_epoch.value == 3

    iterator = iter(dataset)
    assert prepared_epochs == []
    next(iterator)
    assert prepared_epochs == [3]


def test_naflex_persistent_workers_read_shared_epoch():
    dataset = _create_naflex_dataset()
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        persistent_workers=True,
    )

    try:
        epoch_0_indices = _loader_indices(loader)
        dataset.set_epoch(1)
        epoch_1_indices = _loader_indices(loader)

        expected_epoch_0 = [
            index
            for _, _, indices in dataset._prepare_epoch_batches(0)
            for index in indices
        ]
        expected_epoch_1 = [
            index
            for _, _, indices in dataset._prepare_epoch_batches(1)
            for index in indices
        ]
        assert epoch_0_indices == expected_epoch_0
        assert epoch_1_indices == expected_epoch_1
        assert epoch_1_indices != epoch_0_indices
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
