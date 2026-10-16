import pytest
import torch
import torch.nn as nn

from timm.task import (
    ClassificationTask, FeatureDistillationTask, LogitDistillationTask, TokenDistillationTask,
    DistillationTeacher,
    load_task_ema_checkpoint, resume_task_checkpoint,
)
from timm.optim import create_optimizer_v2
from timm.utils import CheckpointSaver


class TinyClassifier(nn.Module):
    def __init__(self, in_chans=3, num_classes=3, hidden=5):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_features = hidden
        self.pretrained_cfg = {'mean': (0.5, 0.5, 0.5), 'std': (0.25, 0.25, 0.25)}
        self.fc = nn.Linear(in_chans, num_classes)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


class TinyFeatureModel(nn.Module):
    def __init__(self, in_chans=3, num_classes=3, hidden=5):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_features = hidden
        self.pretrained_cfg = {'mean': (0.5, 0.5, 0.5), 'std': (0.25, 0.25, 0.25)}
        self.stem = nn.Linear(in_chans, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward_features(self, x):
        return self.stem(x.mean((2, 3)))

    def forward_head(self, x, pre_logits=False):
        return x if pre_logits else self.head(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def no_weight_decay(self):
        return {'stem.weight'}

    def group_matcher(self, coarse=False):
        def _matcher(name):
            return 1 if name.startswith('head') else 0
        return _matcher


class ModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _param_group_weight_decay(optimizer):
    return {
        id(param): group['weight_decay']
        for group in optimizer.param_groups
        for param in group['params']
    }


def _create_tiny_distilled_student():
    from timm import create_model

    return create_model(
        'deit_tiny_distilled_patch16_224', pretrained=False,
        img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, num_classes=3,
    )


def test_task_to_keeps_device_dtype_in_sync():
    task = ClassificationTask(TinyClassifier(), verbose=False).to(dtype=torch.float64)

    # moves that don't specify dtype must keep the current dtype
    for args, kwargs in [(('cpu',), {}), ((), {}), ((), {'memory_format': torch.channels_last})]:
        assert task.to(*args, **kwargs) is task
        assert task.dtype == next(task.parameters()).dtype == torch.float64

    # dtype-only move must keep the current device
    task.to('meta').to(torch.float32)
    assert task.device == next(task.parameters()).device == torch.device('meta')
    assert task.dtype == next(task.parameters()).dtype == torch.float32

    # rejected conversion leaves metadata untouched
    with pytest.raises(TypeError):
        task.to(torch.int64)
    assert task.dtype == torch.float32


@pytest.mark.parametrize('task_cls', [LogitDistillationTask, FeatureDistillationTask, TokenDistillationTask])
def test_distillation_train_keeps_teacher_in_eval(task_cls):
    if task_cls is TokenDistillationTask:
        student = _create_tiny_distilled_student()
    else:
        student = TinyFeatureModel(hidden=4)
    teacher = TinyFeatureModel(hidden=6)
    teacher.stem = nn.Sequential(teacher.stem, nn.BatchNorm1d(6), nn.Dropout(0.5))
    task = task_cls(student, teacher, criterion=nn.CrossEntropyLoss(), verbose=False)
    teacher_buffers = {name: value.clone() for name, value in teacher.named_buffers()}
    x = torch.randn(4, 3, 8, 8)
    target = torch.tensor([0, 1, 2, 0])

    for mode in (True, False, True):
        assert (task.train() if mode else task.eval()) is task
        assert task.training is mode
        assert all(module.training is mode for module in task.get_trainable_module().modules())
        assert task.criterion.training is mode
        assert all(not module.training for module in task.teacher.modules())
        if mode:
            task.zero_grad()
            result = task(x, target)
            result['loss'].backward()
            assert student.head.weight.grad is not None
            if task_cls is TokenDistillationTask:
                assert student.head_dist.weight.grad is not None
            elif task_cls is FeatureDistillationTask:
                assert task.get_trainable_module().projection.weight.grad is not None
            assert all(param.grad is None for param in teacher.parameters())
            for name, value in teacher.named_buffers():
                assert torch.equal(value, teacher_buffers[name]), name


@pytest.mark.parametrize('compiled', [
    False,
    pytest.param(True, marks=pytest.mark.skipif(not hasattr(torch, 'compile'), reason='requires torch.compile')),
])
def test_token_distillation_reuses_teacher_wrapper(compiled):
    teacher = DistillationTeacher(TinyFeatureModel())
    student = _create_tiny_distilled_student()
    task = TokenDistillationTask(student, teacher, verbose=False).train()

    assert task.teacher is teacher
    x = torch.randn(4, 3, 8, 8)
    target = torch.tensor([0, 1, 2, 0])
    with torch.no_grad():
        expected = task(x, target)
    if compiled:
        task.compile(backend='eager')
        task.train()
        assert hasattr(teacher.model, '_orig_mod')

    result = task(x, target)
    for key, value in expected.items():
        torch.testing.assert_close(result[key], value)
    result['loss'].backward()
    assert student.head.weight.grad is not None
    assert student.head_dist.weight.grad is not None
    assert all(param.grad is None for param in teacher.parameters())
    assert all(not module.training for module in teacher.modules())


def test_token_distillation_named_teacher_checkpoint(tmp_path):
    from timm import create_model

    teacher_name = 'test_vit'
    source = create_model(teacher_name, pretrained=False, num_classes=3)
    checkpoint_path = tmp_path / 'teacher.pth'
    torch.save(source.state_dict(), checkpoint_path)
    overlay = dict(mean=(0.5,) * 3, std=(0.25,) * 3)

    task = TokenDistillationTask(
        _create_tiny_distilled_student(),
        teacher_name,
        teacher_pretrained_path=str(checkpoint_path),
        teacher_pretrained_cfg_overlay=overlay,
        verbose=False,
    )

    teacher_state = task.teacher.model.state_dict()
    for key, value in source.state_dict().items():
        assert torch.equal(teacher_state[key], value), key
    torch.testing.assert_close(task.teacher.mean_kd.flatten(), torch.tensor(overlay['mean']))
    torch.testing.assert_close(task.teacher.std_kd.flatten(), torch.tensor(overlay['std']))


def test_task_checkpoint_omits_empty_task_state_and_keeps_legacy_paths(tmp_path):
    task = ClassificationTask(TinyClassifier(), nn.CrossEntropyLoss(), verbose=False)
    optimizer = torch.optim.SGD(task.get_trainable_module().parameters(), lr=0.1)
    task.setup_ema(decay=0.0)

    path = tmp_path / 'task.pth.tar'
    CheckpointSaver(task.get_trainable_module(), optimizer, checkpoint_dir=tmp_path, task=task)._save(path, epoch=2)

    checkpoint = _load(path)
    assert 'state_dict' in checkpoint
    assert 'state_dict_ema' in checkpoint
    assert 'task_state' not in checkpoint
    assert 'task_state_ema' not in checkpoint

    legacy_save_path = tmp_path / 'legacy-save.pth.tar'
    CheckpointSaver(
        task.get_trainable_module(),
        optimizer,
        model_ema=task.get_trainable_module(ema=True),
        checkpoint_dir=tmp_path,
    )._save(legacy_save_path, epoch=2)
    legacy_checkpoint = _load(legacy_save_path)
    assert 'state_dict' in legacy_checkpoint
    assert 'state_dict_ema' in legacy_checkpoint

    legacy_resume_path = tmp_path / 'legacy-resume.pth.tar'
    torch.save({'model': task.get_eval_model().state_dict(), 'epoch': 5, 'version': 2}, legacy_resume_path)
    fresh_task = ClassificationTask(TinyClassifier(), nn.CrossEntropyLoss(), verbose=False)
    assert resume_task_checkpoint(fresh_task, legacy_resume_path, log_info=False) == 6


def test_feature_distillation_checkpoint_keeps_projection_in_task_state(tmp_path):
    task = FeatureDistillationTask(
        TinyFeatureModel(hidden=4),
        TinyFeatureModel(hidden=6),
        criterion=nn.CrossEntropyLoss(),
        verbose=False,
    )
    optimizer = torch.optim.SGD(task.get_trainable_module().parameters(), lr=0.1)
    task.setup_ema(decay=0.0)

    path = tmp_path / 'fd.pth.tar'
    CheckpointSaver(task.get_trainable_module(), optimizer, checkpoint_dir=tmp_path, task=task)._save(path, epoch=1)
    checkpoint = _load(path)

    assert 'projection' in checkpoint['task_state']
    assert 'projection' in checkpoint['task_state_ema']
    assert not any(k.startswith('projection') for k in checkpoint['state_dict'])
    assert not any(k.startswith('projection') for k in checkpoint['state_dict_ema'])
    assert task.get_eval_model()(torch.randn(2, 3, 8, 8)).shape == (2, 3)
    assert task.get_eval_model(ema=True)(torch.randn(2, 3, 8, 8)).shape == (2, 3)

    wrapped_trainable = ModuleWrapper(task.get_trainable_module())
    assert task.get_eval_model(module=wrapped_trainable) is wrapped_trainable.module.student

    fresh_task = FeatureDistillationTask(
        TinyFeatureModel(hidden=4),
        TinyFeatureModel(hidden=6),
        criterion=nn.CrossEntropyLoss(),
        verbose=False,
    )
    fresh_task.setup_ema(decay=0.0)
    resume_task_checkpoint(fresh_task, path, log_info=False, weights_only=False)
    load_task_ema_checkpoint(fresh_task, path, weights_only=False)

    matcher = task.get_trainable_module().group_matcher()
    assert matcher('projection.weight') == matcher('student.head.weight')


def test_feature_distillation_trainable_module_optimizer_grouping():
    task = FeatureDistillationTask(
        TinyFeatureModel(hidden=4),
        TinyFeatureModel(hidden=6),
        criterion=nn.CrossEntropyLoss(),
        verbose=False,
    )
    trainable_module = task.get_trainable_module()
    optimizer = create_optimizer_v2(
        trainable_module,
        opt='sgd',
        lr=0.1,
        weight_decay=0.2,
    )
    weight_decay_by_param = _param_group_weight_decay(optimizer)

    assert id(trainable_module.projection.weight) in weight_decay_by_param
    assert id(trainable_module.projection.bias) in weight_decay_by_param
    assert weight_decay_by_param[id(trainable_module.projection.weight)] == 0.2
    assert weight_decay_by_param[id(trainable_module.projection.bias)] == 0.0
    assert weight_decay_by_param[id(trainable_module.student.stem.weight)] == 0.0

    matcher = trainable_module.group_matcher()
    assert matcher('projection.weight') == matcher('student.head.weight')


def test_base_eval_model_preserves_trainable_wrapper_for_use():
    task = ClassificationTask(TinyClassifier(), nn.CrossEntropyLoss(), verbose=False)
    wrapper = ModuleWrapper(task.get_trainable_module())
    task.trainable_module = wrapper

    assert task.get_eval_model() is wrapper
    assert set(task.get_clip_parameters()) == set(wrapper.parameters())


@pytest.mark.skipif(not hasattr(torch, 'compile'), reason='requires torch.compile')
def test_compiled_ema_eval_model_reflects_update():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 8, 8)

    task = ClassificationTask(TinyClassifier(), nn.CrossEntropyLoss(), verbose=False)
    task.setup_ema(decay=0.0)
    task.compile(backend='eager')
    task.compile_ema(backend='eager')
    assert hasattr(task.get_eval_model(), '_orig_mod')
    assert hasattr(task.get_eval_model(ema=True), '_orig_mod')

    before = task.get_eval_model(ema=True)(x).detach().clone()
    with torch.no_grad():
        for p in task.get_trainable_module().parameters():
            p.add_(0.25)

    task.update_ema(step=2)
    after = task.get_eval_model(ema=True)(x).detach()
    current = task.get_eval_model()(x).detach()

    assert not torch.allclose(before, after)
    assert torch.allclose(after, current)


@pytest.mark.skipif(not hasattr(torch, 'compile'), reason='requires torch.compile')
def test_compiled_eval_checkpoint_load_uses_unwrapped_state_dict_target():
    src_task = ClassificationTask(TinyClassifier(), nn.CrossEntropyLoss(), verbose=False)
    state_dict = src_task.get_checkpoint_state()['state_dict']

    task = ClassificationTask(TinyClassifier(), nn.CrossEntropyLoss(), verbose=False)
    task.compile(backend='eager')
    assert hasattr(task.get_eval_model(), '_orig_mod')

    task.load_checkpoint_state(state_dict)


@pytest.fixture(params=['classification', 'logit', 'feature'])
def conversion_task(request):
    if request.param == 'classification':
        return ClassificationTask(TinyClassifier(), verbose=False)
    student, teacher = TinyFeatureModel(hidden=4), TinyFeatureModel(hidden=6)
    task_cls = LogitDistillationTask if request.param == 'logit' else FeatureDistillationTask
    return task_cls(student, teacher, criterion=nn.CrossEntropyLoss(), verbose=False)


@pytest.mark.parametrize('conversion,initial_dtype,expected_dtype', [
    ('double', torch.float32, torch.float64),
    ('float', torch.float64, torch.float32),
    ('half', torch.float32, torch.float16),
    ('bfloat16', torch.float32, torch.bfloat16),
    ('type', torch.float32, torch.float64),
    ('parent_to', torch.float32, torch.float64),
    ('to_dtype', torch.float64, torch.float32),
    ('to_device', torch.float64, torch.float64),
    ('memory_format', torch.float64, torch.float64),
])
def test_task_module_conversion_keeps_metadata(conversion_task, conversion, initial_dtype, expected_dtype):
    task = conversion_task.to(dtype=initial_dtype)
    if conversion == 'type':
        result = task.type(torch.DoubleTensor)
    elif conversion == 'parent_to':
        parent = ModuleWrapper(task)
        assert parent.to(dtype=expected_dtype) is parent
        result = task
    elif conversion == 'to_dtype':
        result = task.to(dtype=expected_dtype)
    elif conversion == 'to_device':
        result = task.to('cpu')
    elif conversion == 'memory_format':
        result = task.to(memory_format=torch.channels_last)
    else:
        result = getattr(task, conversion)()
    assert result is task
    assert task.dtype == next(task.parameters()).dtype == expected_dtype
    assert task.device == next(task.parameters()).device == torch.device('cpu')
    if expected_dtype in (torch.float32, torch.float64):
        batch = torch.randn(4, 3, 8, 8, device=task.device, dtype=task.dtype)
        output = task(batch, torch.tensor([0, 1, 2, 0]))
        assert torch.isfinite(output['loss'])
        output['loss'].backward()
        assert any(p.grad is not None for p in task.get_trainable_module().parameters())


def test_task_parent_device_conversion_keeps_metadata(conversion_task):
    task = conversion_task
    ModuleWrapper(task).to('meta')
    assert task.device == next(task.parameters()).device == torch.device('meta')


@pytest.mark.parametrize('initial_dtype,method,expected_dtype', [
    (torch.float32, 'double', torch.float64),
    (torch.float64, 'float', torch.float32),
    (torch.complex64, 'double', torch.complex64),
    (torch.complex128, 'float', torch.complex128),
])
def test_parameterless_task_conversion_metadata(initial_dtype, method, expected_dtype):
    from timm.task.task import TrainingTask

    task = TrainingTask(dtype=initial_dtype, verbose=False)
    assert getattr(task, method)() is task
    assert task.dtype == expected_dtype
    assert not list(task.parameters())
    assert not list(task.buffers())


@pytest.mark.parametrize('parent_conversion', [False, True])
def test_task_converted_resnet_training_batch(parent_conversion):
    from timm import create_model

    task = ClassificationTask(create_model('resnet18', pretrained=False, num_classes=3), verbose=False)
    if parent_conversion:
        ModuleWrapper(task).to(dtype=torch.float64)
    else:
        task.double()
    batch = torch.randn(2, 3, 32, 32, device=task.device, dtype=task.dtype)
    result = task(batch, torch.tensor([0, 2]))
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert task.get_trainable_module().conv1.weight.grad is not None


@pytest.mark.parametrize('parent_conversion', [False, True])
def test_task_memory_format_keeps_metadata(parent_conversion):
    from timm import create_model

    task = ClassificationTask(create_model('resnet18', pretrained=False, num_classes=3), verbose=False)
    task.to(dtype=torch.float64)
    converted = ModuleWrapper(task) if parent_conversion else task
    converted.to(memory_format=torch.channels_last)
    assert task.dtype == next(task.parameters()).dtype == torch.float64
    assert task.get_trainable_module().conv1.weight.is_contiguous(memory_format=torch.channels_last)
    batch = torch.randn(2, 3, 32, 32, device=task.device, dtype=task.dtype)
    result = task(batch, torch.tensor([0, 2]))
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert task.get_trainable_module().conv1.weight.grad is not None
