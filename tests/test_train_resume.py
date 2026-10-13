"""End-to-end tests of train.py's stop/milestone/in-place resume behaviour and YAML config validation."""
import csv
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from timm.utils import load_checkpoint_metadata

# torch < 2.5 (no safe_globals) can't load training checkpoints (floats, argparse.Namespace) with weights_only=True,
# nor resume them
needs_checkpoint_load = pytest.mark.skipif(
    not hasattr(torch.serialization, 'safe_globals'), reason='requires torch.load weights_only w/ safe_globals')


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(3, num_classes)
        self.pretrained_cfg = dict(input_size=(3, 8, 8), mean=(0.5,) * 3, std=(0.5,) * 3,
                                   interpolation='bilinear', crop_pct=1.)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


@pytest.fixture
def image_folder(tmp_path):
    """A two class ImageFolder dataset with train and validation splits."""
    root = tmp_path / 'data'
    for split in ('train', 'validation'):
        for cls, color in (('a', (200, 30, 30)), ('b', (30, 30, 200))):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                Image.new('RGB', (8, 8), color=color).save(d / f'{i}.png')
    return root


def _train_args(image_folder, output, **overrides):
    args = dict(
        data_dir=str(image_folder), num_classes=2, device='cpu', workers=0, batch_size=4, img_size=8,
        opt='sgd', lr=0.01, warmup_epochs=0, no_aug=True, output=str(output), checkpoint_hist=2,
    )
    args.update(overrides)
    argv = ['train.py']
    for k, v in args.items():
        flag = '--' + k.replace('_', '-')
        if v is True:
            argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv.extend([flag, *map(str, v)])
        else:
            argv.extend([flag, str(v)])
    return argv


def _run_train(monkeypatch, argv):
    import train
    monkeypatch.setattr(train, 'create_model', lambda *a, **kw: _TinyClassifier(kw['num_classes']))
    monkeypatch.setattr('sys.argv', argv)
    train.main()


def _summary_epochs(exp_dir):
    with (exp_dir / 'summary.csv').open() as f:
        lines = [l for l in f.read().splitlines() if l]
    assert sum(l.startswith('epoch') for l in lines) == 1, 'summary should have exactly one header'
    return [int(r['epoch']) for r in csv.DictReader(lines)]


def _checkpoint_epoch(path):
    return load_checkpoint_metadata(path)['epoch']


@needs_checkpoint_load
@pytest.mark.usefixtures('isolate_cli_backend_flags')
def test_stop_epoch_and_milestones(monkeypatch, tmp_path, image_folder):
    exp_dir = tmp_path / 'stop'
    _run_train(monkeypatch, _train_args(
        image_folder, tmp_path, experiment='stop', epochs=4, stop_epoch=2, save_epochs=[1, 2], val_interval=2,
    ))
    # the milestone at 1 completed epoch forces an eval the val interval would skip, the stop epoch is always evaluated
    assert _summary_epochs(exp_dir) == [0, 1]
    assert (exp_dir / 'milestone-1.pth.tar').exists() and (exp_dir / 'milestone-2.pth.tar').exists()
    assert _checkpoint_epoch(exp_dir / 'milestone-2.pth.tar') == 1
    assert _checkpoint_epoch(exp_dir / 'last.pth.tar') == 1
    args = yaml.safe_load((exp_dir / 'args.yaml').read_text())
    assert args['epochs'] == 4 and args['stop_epoch'] == 2


@pytest.mark.usefixtures('isolate_cli_backend_flags')
def test_stop_epoch_out_of_range(monkeypatch, tmp_path, image_folder):
    with pytest.raises(ValueError, match='stop-epoch'):
        _run_train(monkeypatch, _train_args(image_folder, tmp_path, experiment='bad', epochs=2, stop_epoch=3))


@needs_checkpoint_load
@pytest.mark.usefixtures('isolate_cli_backend_flags')
def test_resume_auto_in_place(monkeypatch, tmp_path, image_folder):
    exp_dir = tmp_path / 'auto'
    base = dict(experiment='auto', epochs=4, resume='auto', checkpoint_hist=2)
    # no checkpoint yet, auto starts a fresh run and stops after 2 epochs
    _run_train(monkeypatch, _train_args(image_folder, tmp_path, stop_epoch=2, **base))
    assert _summary_epochs(exp_dir) == [0, 1]
    assert _checkpoint_epoch(exp_dir / 'last.pth.tar') == 1

    # same command without the stop, picks up from last.pth.tar and runs the remaining epochs in the same dir
    _run_train(monkeypatch, _train_args(image_folder, tmp_path, **base))
    assert _summary_epochs(exp_dir) == [0, 1, 2, 3]
    assert _checkpoint_epoch(exp_dir / 'last.pth.tar') == 3
    # the ranked history was restored so the max_history limit still holds across the resume
    assert len(list(exp_dir.glob('checkpoint-*.pth.tar'))) == 2
    # model_best tracks the best metric over the whole run, not just the epochs after the resume
    with (exp_dir / 'summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    best = max(range(len(rows)), key=lambda i: (float(rows[i]['eval_top1']), -i))
    assert _checkpoint_epoch(exp_dir / 'model_best.pth.tar') == best
    # args.yaml records the command as given, so re-running it is idempotent
    assert yaml.safe_load((exp_dir / 'args.yaml').read_text())['resume'] == 'auto'

    # already complete, a further auto resume is a no-op that doesn't disturb the summary
    _run_train(monkeypatch, _train_args(image_folder, tmp_path, **base))
    assert _summary_epochs(exp_dir) == [0, 1, 2, 3]


@pytest.mark.usefixtures('isolate_cli_backend_flags')
def test_resume_auto_requires_experiment(monkeypatch, tmp_path, image_folder):
    with pytest.raises(SystemExit):
        _run_train(monkeypatch, _train_args(image_folder, tmp_path, epochs=1, resume='auto'))


@pytest.mark.usefixtures('isolate_cli_backend_flags')
def test_fresh_run_resets_existing_summary(monkeypatch, tmp_path, image_folder):
    exp_dir = tmp_path / 'fresh'
    _run_train(monkeypatch, _train_args(image_folder, tmp_path, experiment='fresh', epochs=2))
    assert _summary_epochs(exp_dir) == [0, 1]
    _run_train(monkeypatch, _train_args(image_folder, tmp_path, experiment='fresh', epochs=1))
    assert _summary_epochs(exp_dir) == [0]


@pytest.fixture
def restore_parser_defaults():
    # --config applies the file through parser.set_defaults(), which mutates the module-level parser for later tests
    import train
    saved = dict(train.parser._defaults), [(a, a.default) for a in train.parser._actions]
    yield
    train.parser._defaults = saved[0]
    for action, default in saved[1]:
        action.default = default


def _parse_with_config(monkeypatch, tmp_path, cfg):
    import train
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr('sys.argv', ['train.py', '--config', str(path), '--data-dir', '/nonexistent'])
    return train._parse_args()[0]


@pytest.mark.usefixtures('restore_parser_defaults')
def test_load_config_warns_on_unknown_key(monkeypatch, tmp_path, caplog):
    args = _parse_with_config(monkeypatch, tmp_path, dict(lr=0.5, not_an_arg=1))
    assert args.lr == 0.5
    assert 'not_an_arg' in caplog.text


@pytest.mark.usefixtures('restore_parser_defaults')
@pytest.mark.parametrize('cfg', [
    dict(train_size_schedule='nope'),  # invalid choice
    dict(lr='fast'),              # non-numeric string where a float is expected
    dict(input_size=224),         # scalar where a list is expected
    dict(decay_milestones=[1, 'x']),
])
def test_load_config_rejects_invalid_values(monkeypatch, tmp_path, cfg):
    with pytest.raises(SystemExit):
        _parse_with_config(monkeypatch, tmp_path, cfg)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_load_config_accepts_valid_values(monkeypatch, tmp_path):
    args = _parse_with_config(monkeypatch, tmp_path, dict(
        sched='wsd', lr=3e-4, epochs=10, input_size=[3, 32, 32], mean=[0.5], amp=True, model_ema=True,
        pretrained_cfg_overlay={'num_classes': 10}, save_epochs=[4, 8], stop_epoch=None,
    ))
    assert args.sched == 'wsd' and args.input_size == [3, 32, 32] and args.amp and args.save_epochs == [4, 8]


@pytest.mark.usefixtures('restore_parser_defaults')
def test_load_config_converts_numeric_strings(monkeypatch, tmp_path):
    # YAML 1.1 reads 3e-4 as a string, the config loader converts it like the command line would
    args = _parse_with_config(monkeypatch, tmp_path, dict(lr='3e-4', epochs='10', mean=['0.5', 0.4, '0.3']))
    assert args.lr == 3e-4 and args.epochs == 10 and args.mean == [0.5, 0.4, 0.3]
