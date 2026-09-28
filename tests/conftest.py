from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def isolate_cli_backend_flags(monkeypatch):
    # train.py / validate.py enable TF32 and cudnn.benchmark globally when CUDA is available. Give them
    # stand-ins so in-process CLI runs don't leak into other tests (or trip disable_global_flags).
    monkeypatch.setattr(torch.backends.cuda, 'matmul', SimpleNamespace(allow_tf32=False))
    monkeypatch.setattr(torch.backends, 'cudnn', SimpleNamespace(benchmark=False))
