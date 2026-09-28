from pathlib import Path

import pytest
import torch

from timm.utils.checkpoint_saver import CheckpointSaver


def _create_saver(checkpoint_dir: Path, max_history: int = 2, decreasing: bool = False) -> CheckpointSaver:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return CheckpointSaver(
        model,
        optimizer,
        checkpoint_dir=str(checkpoint_dir),
        max_history=max_history,
        decreasing=decreasing,
    )


@pytest.mark.parametrize('max_history', [2, 3])
def test_save_checkpoint_without_metric_retains_latest_history(tmp_path: Path, max_history):
    saver = _create_saver(tmp_path, max_history=max_history)

    for epoch in range(max_history + 1):
        assert saver.save_checkpoint(epoch) == (None, None)

    assert [(Path(path).name, metric) for path, metric in saver.checkpoint_files] == [
        (f'checkpoint-{epoch}.pth.tar', None) for epoch in range(max_history, 0, -1)
    ]
    assert not (tmp_path / 'checkpoint-0.pth.tar').exists()
    for epoch in range(1, max_history + 1):
        assert (tmp_path / f'checkpoint-{epoch}.pth.tar').exists()


def test_save_checkpoint_replaces_unranked_history_with_ranked(tmp_path: Path):
    saver = _create_saver(tmp_path)

    saver.save_checkpoint(0, metric=0.8)
    saver.save_checkpoint(1)
    assert saver.save_checkpoint(2, metric=0.9) == (0.9, 2)

    assert [metric for _, metric in saver.checkpoint_files] == [0.9, 0.8]
    assert not (tmp_path / 'checkpoint-1.pth.tar').exists()


def test_save_checkpoint_keeps_unranked_order_with_ranked_history(tmp_path: Path):
    saver = _create_saver(tmp_path, max_history=4)

    saver.save_checkpoint(0, metric=0.8)
    for epoch in range(1, 5):
        saver.save_checkpoint(epoch)

    assert [Path(path).name for path, _ in saver.checkpoint_files] == [
        'checkpoint-0.pth.tar', 'checkpoint-4.pth.tar', 'checkpoint-3.pth.tar', 'checkpoint-2.pth.tar'
    ]
    assert not (tmp_path / 'checkpoint-1.pth.tar').exists()


@pytest.mark.parametrize('decreasing, ranked_metrics', [(False, (0.9, 0.8)), (True, (0.1, 0.2))])
def test_save_checkpoint_does_not_replace_ranked_history_with_unranked(tmp_path: Path, decreasing, ranked_metrics):
    saver = _create_saver(tmp_path, decreasing=decreasing)

    for epoch, metric in enumerate(ranked_metrics):
        saver.save_checkpoint(epoch, metric=metric)

    assert saver.save_checkpoint(2) == (ranked_metrics[0], 0)
    assert [metric for _, metric in saver.checkpoint_files] == list(ranked_metrics)
    assert (tmp_path / 'checkpoint-0.pth.tar').exists()
    assert (tmp_path / 'checkpoint-1.pth.tar').exists()
    assert not (tmp_path / 'checkpoint-2.pth.tar').exists()


def test_save_checkpoint_tie_keeps_earlier_epoch_first(tmp_path: Path):
    saver = _create_saver(tmp_path)

    saver.save_checkpoint(0, metric=0.9)
    assert saver.save_checkpoint(1, metric=0.9) == (0.9, 0)
    assert [Path(path).name for path, _ in saver.checkpoint_files] == [
        'checkpoint-0.pth.tar', 'checkpoint-1.pth.tar'
    ]

    assert saver.save_checkpoint(2, metric=0.95) == (0.95, 2)
    assert [Path(path).name for path, _ in saver.checkpoint_files] == [
        'checkpoint-2.pth.tar', 'checkpoint-0.pth.tar'
    ]
    assert not (tmp_path / 'checkpoint-1.pth.tar').exists()
