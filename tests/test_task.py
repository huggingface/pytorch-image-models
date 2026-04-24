import pytest
import torch
import torch.nn as nn

from timm.task import ClassificationTask, FeatureDistillationTask, load_task_ema_checkpoint, resume_task_checkpoint
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
