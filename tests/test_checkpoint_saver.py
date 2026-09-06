from pathlib import Path

import torch

from timm.utils.checkpoint_saver import CheckpointSaver


def _create_saver(checkpoint_dir: Path, max_history: int = 2) -> CheckpointSaver:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return CheckpointSaver(
        model,
        optimizer,
        checkpoint_dir=str(checkpoint_dir),
        max_history=max_history,
    )


def test_save_checkpoint_without_metric_retains_latest_history(tmp_path: Path):
    saver = _create_saver(tmp_path)

    for epoch in range(3):
        assert saver.save_checkpoint(epoch) == (None, None)

    assert [(Path(path).name, metric) for path, metric in saver.checkpoint_files] == [
        ('checkpoint-2.pth.tar', None),
        ('checkpoint-1.pth.tar', None),
    ]
    assert not (tmp_path / 'checkpoint-0.pth.tar').exists()
    assert (tmp_path / 'checkpoint-1.pth.tar').exists()
    assert (tmp_path / 'checkpoint-2.pth.tar').exists()


def test_save_checkpoint_replaces_unranked_history_with_ranked(tmp_path: Path):
    saver = _create_saver(tmp_path)

    saver.save_checkpoint(0, metric=0.8)
    saver.save_checkpoint(1)
    assert saver.save_checkpoint(2, metric=0.9) == (0.9, 2)

    assert [metric for _, metric in saver.checkpoint_files] == [0.9, 0.8]
    assert not (tmp_path / 'checkpoint-1.pth.tar').exists()
