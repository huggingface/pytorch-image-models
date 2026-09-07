import subprocess
import sys

import torch
from PIL import Image

import timm
from timm.utils import CheckpointSaver


def test_inference_defaults_to_registered_test_input_size(tmp_path):
    data_dir = tmp_path / "images" / "class0"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (301, 247), color=(30, 100, 220)).save(data_dir / "sample.png")

    model_name = "resnet10t.c3_in1k"
    model = timm.create_model(model_name, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    CheckpointSaver(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=str(checkpoint_dir),
        recovery_dir=str(checkpoint_dir),
    ).save_checkpoint(epoch=0)

    results_dir = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            "inference.py",
            "--data-dir",
            str(data_dir.parent),
            "--model",
            model_name,
            "--checkpoint",
            str(checkpoint_dir / "last.pth.tar"),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--batch-size",
            "1",
            "--results-dir",
            str(results_dir),
            "--no-console-results",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert sorted(path.name for path in results_dir.iterdir()) == ["resnet10t.c3_in1k-224.csv"]
