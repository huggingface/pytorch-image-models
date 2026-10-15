from copy import deepcopy

import pytest
import torch
from torch import nn

from timm.task import ClassificationTask, LogitDistillationTask, MultiLabelClassificationTask


def make_task(task_kind):
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(4, 3))
    model.num_classes = 3
    model.in_chans = 3
    model.num_features = 4
    model.pretrained_cfg = {'mean': (0.5, 0.5, 0.5), 'std': (0.25, 0.25, 0.25)}
    if task_kind == 'distillation':
        return LogitDistillationTask(model, deepcopy(model), verbose=False)
    if task_kind == 'multilabel':
        return MultiLabelClassificationTask(model, verbose=False)
    return ClassificationTask(model, verbose=False)


@pytest.mark.parametrize('task_kind', ['classification', 'multilabel', 'distillation'])
@pytest.mark.parametrize('move', ['device', 'non_blocking', 'no_args', 'channels_last', 'other_tensor', 'dtype'])
def test_task_to_keeps_metadata_and_forward_in_sync(task_kind, move):
    task = make_task(task_kind).to(dtype=torch.float64)
    expected_dtype = torch.float32 if move == 'dtype' else torch.float64
    if move == 'device':
        result = task.to('cpu')
    elif move == 'non_blocking':
        result = task.to(device='cpu', non_blocking=True)
    elif move == 'no_args':
        result = task.to()
    elif move == 'channels_last':
        result = task.to(memory_format=torch.channels_last)
        assert task.get_trainable_module()[0].weight.is_contiguous(memory_format=torch.channels_last)
    elif move == 'other_tensor':
        result = task.to(torch.empty(0, dtype=expected_dtype))
    else:
        result = task.to(expected_dtype)
    assert result is task
    assert task.device == next(task.parameters()).device == torch.device('cpu')
    assert task.dtype == next(task.parameters()).dtype == expected_dtype
    target = (
        torch.tensor([[1., 0., 1.], [0., 1., 0.]], dtype=task.dtype)
        if task_kind == 'multilabel' else torch.tensor([0, 2])
    )
    output = task(torch.randn(2, 3, 4, 4, dtype=task.dtype), target)
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    assert task.get_trainable_module()[0].weight.grad is not None


def test_task_dtype_only_move_preserves_meta_device():
    task = ClassificationTask(nn.Linear(2, 3), verbose=False).to('meta')
    task.to(dtype=torch.float64)
    assert task.device == next(task.parameters()).device == torch.device('meta')
    assert task.dtype == next(task.parameters()).dtype == torch.float64


@pytest.mark.parametrize('task_kind', ['classification', 'multilabel', 'distillation'])
def test_task_rejected_dtype_does_not_change_metadata(task_kind):
    task = make_task(task_kind).to(dtype=torch.float64)
    before = task.device, task.dtype
    with pytest.raises(TypeError):
        task.to(dtype=torch.int64)
    assert (task.device, task.dtype) == before
    assert next(task.parameters()).dtype == torch.float64
