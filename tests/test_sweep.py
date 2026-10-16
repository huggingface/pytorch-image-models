"""Tests for sweep.py: search space expansion, plan generation, WSD branching and the local executor."""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

import sweep

if sys.platform == 'win32':
    pytest.skip('sweep.py requires Unix flock; on Windows use WSL2', allow_module_level=True)

pytestmark = pytest.mark.skipif(not sweep.TORCH_SUPPORTED, reason=sweep.TORCH_REQUIREMENT)


@pytest.fixture
def restore_parser_defaults():
    import train
    saved = dict(train.parser._defaults), [(a, a.default) for a in train.parser._actions]
    yield
    train.parser._defaults = saved[0]
    for action, default in saved[1]:
        action.default = default


def _spec(tmp_path, **overrides):
    spec = dict(
        name='test',
        output=str(tmp_path / 'sweeps'),
        train=dict(model='test_vit', data_dir='/data', num_classes=2, warmup_epochs=1, epochs=10),
        search=dict(mode='grid', space=dict(lr=[0.1, 0.2])),
    )
    spec.update(overrides)
    return spec


# a train block without epochs, which WSD sweeps derive per job
WSD_TRAIN = dict(model='test_vit', data_dir='/data', num_classes=2, warmup_epochs=1)


def _write(spec, force=False):
    """Plan and write a sweep, returning its root and the plan as loaded back from disk."""
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec, force=force))
    return root, sweep.load_plan(root)


def _mark_all(root, plan, status):
    for job in plan['jobs']:
        sweep.JobState(str(root), job).set(status)


def _start_attempt(state, attempt_id, **attempt_update):
    """Record a started attempt as Executor.launch does, with no lock held: its worker is gone."""
    for path in (state.coord_path, state.train_path):
        lock = sweep.Lock(path)
        assert lock.acquire(exclusive=True)
        lock.release()
    state.start_attempt(attempt_id, [], ['run.sh'])
    if attempt_update:
        state.set('running', **attempt_update)
    state.set_live_attempt(attempt_id)


def _join(state, attempt_id):
    """Start a rank of `attempt_id` as train.py does. Returns its lock descriptors, held until `_exit`."""
    import train
    joined = False
    try:
        train._join_sweep_attempt([sweep.plan_lock(state.root).path, state.train_path], state.attempt_path, attempt_id)
        joined = True
    finally:
        fds = list(train._sweep_lock_fds)
        train._sweep_lock_fds.clear()
        if not joined:
            _exit(fds)
    return fds


def _exit(fds):
    for fd in fds:
        os.close(fd)


# -- search space

def test_expand_grid_with_zip():
    trials = sweep.expand_search(dict(
        mode='grid', space=dict(lr=[0.1, 0.2], weight_decay={'choice': [0, 1e-4]}),
        zip=[dict(batch_size=[64, 128], lr_base_size=[64, 128])],
    ))
    assert len(trials) == 8
    assert trials[0] == dict(lr=0.1, weight_decay=0, batch_size=64, lr_base_size=64)
    assert all(t['batch_size'] == t['lr_base_size'] for t in trials)


def test_expand_grid_rejects_distributions():
    with pytest.raises(sweep.SweepError, match='distribution'):
        sweep.expand_search(dict(mode='grid', space=dict(lr={'loguniform': [1e-4, 1e-2]})))


def test_expand_random_is_seeded():
    search = dict(
        mode='random', seed=7, n_trials=5,
        space=dict(lr={'loguniform': [1e-4, 1e-2]}, mixup={'uniform': [0, 0.5]}, epochs={'int': [10, 20]},
                   opt={'choice': ['sgd', 'adamw']}),
    )
    a, b = sweep.expand_search(search), sweep.expand_search(search)
    assert a == b and len(a) == 5
    assert all(1e-4 <= t['lr'] <= 1e-2 and 10 <= t['epochs'] <= 20 and t['opt'] in ('sgd', 'adamw') for t in a)


def test_zip_group_lengths_must_match():
    with pytest.raises(sweep.SweepError, match='equal length'):
        sweep.expand_search(dict(mode='grid', zip=[dict(a=[1, 2], b=[1])]))


# -- planning

@pytest.mark.usefixtures('restore_parser_defaults')
def test_plan_plain_sweep(tmp_path):
    plan = sweep.plan_sweep(_spec(tmp_path, search=dict(mode='grid', repeats=2, space=dict(lr=[0.1, 0.2]))))
    assert len(plan['trials']) == 4 and len(plan['jobs']) == 4
    assert [t['id'] for t in plan['trials']] == [
        'trial-000-lr0.1-r0', 'trial-000-lr0.1-r1', 'trial-001-lr0.2-r0', 'trial-001-lr0.2-r1']
    assert [t['params']['seed'] for t in plan['trials']] == [42, 43, 42, 43]
    job = plan['jobs'][0]
    assert job['kind'] == 'train' and job['deps'] == [] and job['end_epoch'] == 10
    cfg = job['_cfg']
    assert cfg['resume'] == 'auto' and cfg['experiment'] == 'train' and cfg['lr'] == 0.1 and cfg['model'] == 'test_vit'
    assert cfg['output'] == str(tmp_path / 'sweeps' / 'test' / 'trial-000-lr0.1-r0')
    assert 'opt' in cfg  # full effective config, parser defaults included
    assert plan['metric'] == {'name': 'top1', 'direction': 'maximize'}


@pytest.mark.usefixtures('restore_parser_defaults')
def test_plan_wsd_sweep(tmp_path):
    plan = sweep.plan_sweep(_spec(
        tmp_path, train=dict(model='test_vit', data_dir='/data', num_classes=2, warmup_epochs=2, cooldown_epochs=1),
        wsd=dict(points=[20, 10], decay_epochs=[2, 4], decay_fn=['1-sqrt', 'linear'], min_lr=[0, 1e-6]),
    ))
    jobs = {j['id']: j for j in plan['jobs']}
    trial = plan['trials'][0]['id']
    # per trial: 2 trunk segments + 2 points x 2 decays x 2 fns x 2 min_lr cooldowns
    assert len(plan['jobs']) == 2 * (2 + 16)
    trunk10, trunk20 = jobs[f'{trial}/stable-10'], jobs[f'{trial}/stable-20']
    assert trunk10['deps'] == [] and trunk20['deps'] == [trunk10['id']]
    assert trunk10['_cfg']['stop_epoch'] == 10 and trunk20['_cfg']['stop_epoch'] == 20
    for trunk in (trunk10, trunk20):
        cfg = trunk['_cfg']
        assert cfg['sched'] == 'wsd' and cfg['sched_on_updates'] and cfg['resume'] == 'auto'
        assert cfg['save_epochs'] == [10, 20] and cfg['experiment'] == 'stable'
        assert cfg['epochs'] == 21 and cfg['decay_epochs'] == 1  # decay sits after the last point, never reached
        assert trunk['config'] != trunk10['config'] or trunk is trunk10
    branch = jobs[f'{trial}/decay-p10-d4-linear-minlr1e-06']
    cfg = branch['_cfg']
    assert branch['deps'] == [trunk10['id']]
    assert branch['inputs'] == [os.path.join(trial, 'stable', 'milestone-10.pth.tar')]
    assert cfg['resume'].endswith(os.path.join(trial, 'stable', 'milestone-10.pth.tar'))
    assert cfg['epochs'] == 14 and cfg['decay_epochs'] == 4 and cfg['lr_decay_fn'] == 'linear' and cfg['min_lr'] == 1e-6
    assert cfg['stop_epoch'] is None and cfg['save_epochs'] == []
    assert branch['end_epoch'] == 10 + 4 + 1  # cooldown epoch included
    assert branch['wsd'] == dict(point=10, decay_epochs=4, decay_fn='linear', min_lr=1e-6)


def test_wsd_warmup_prefix_horizon():
    cfg = dict(warmup_epochs=5, warmup_prefix=True)
    assert sweep.schedule_horizon(40, 8, cfg) == 43  # 5 warmup + 43 = 48 total, decay over the last 8
    cfg['warmup_prefix'] = False
    assert sweep.schedule_horizon(40, 8, cfg) == 48


@pytest.mark.usefixtures('restore_parser_defaults')
@pytest.mark.parametrize('spec_update, match', [
    (dict(train=dict(model='test_vit', output='x')), 'planner'),
    (dict(search=dict(space=dict(lr=[0.1], resume=['auto']))), 'planner'),
    (dict(search=dict(space=dict(lr=[0.1], epochs=[1, 2])), wsd=dict(points=[5])), 'planner'),
    (dict(search=dict(space=dict(not_an_arg=[1, 2]))), 'no train.py argument'),
    (dict(search=dict(space=dict(lr=[0.1], seed=[1, 2]), repeats=2)), 'seed'),
    (dict(train=dict(model='test_vit', warmup_epochs=2), wsd=dict(points=[1])), 'after the warmup'),
    (dict(train=dict(model='test_vit', train_size_schedule='progressive'), wsd=dict(points=[5])), 'progressive'),
    (dict(train=dict(model='test_vit', no_resume_opt=True), wsd=dict(points=[5])), 'optimizer'),
    (dict(train=dict(model='test_vit', sched='cosine'), wsd=dict(points=[5])), 'sched'),
])
def test_plan_rejects_invalid_specs(tmp_path, spec_update, match):
    with pytest.raises(sweep.SweepError, match=match):
        sweep.plan_sweep(_spec(tmp_path, **spec_update))


@pytest.mark.usefixtures('restore_parser_defaults')
def test_plan_uses_base_config_file(tmp_path):
    base = tmp_path / 'base.yaml'
    base.write_text(yaml.safe_dump(dict(model='test_vit', lr='3e-4', opt='adamw')))
    spec = _spec(tmp_path, train=dict(config='base.yaml', weight_decay=0.05))
    plan = sweep.plan_sweep(spec, str(tmp_path))
    cfg = plan['jobs'][0]['_cfg']
    assert cfg['opt'] == 'adamw' and cfg['weight_decay'] == 0.05 and cfg['lr'] == 0.1  # swept lr wins over the base


@pytest.mark.usefixtures('restore_parser_defaults')
def test_write_plan_is_idempotent_and_guards_changes(tmp_path):
    spec = _spec(tmp_path)
    root, plan = _write(spec)
    assert (root / 'plan.json').exists() and (root / 'graph.dot').exists()
    job = plan['jobs'][0]
    assert (root / job['config']).exists() and os.access(root / job['script'], os.X_OK)
    _write(spec)  # same plan, fine
    changed = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.3])))
    with pytest.raises(sweep.SweepError, match='different'):
        _write(changed)
    _write(changed, force=True)
    assert len(sweep.load_plan(root)['jobs']) == 2


@pytest.mark.usefixtures('restore_parser_defaults')
@pytest.mark.parametrize('stop_after', ['config', 'script', 'graph', 'plan'])
@pytest.mark.parametrize('key,value,started', [('weight_decay', 0.2, False), ('checkpoint_hist', 7, True)])
def test_interrupted_plan_publication_and_repair(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_after: str, key: str, value: object, started: bool,
):
    spec = _spec(tmp_path)
    root, original = _write(spec)
    if started:
        _mark_all(root, original, 'completed')
    states = {p: p.read_bytes() for p in (root / 'state').glob('*.json')}
    files = {p for p in root.rglob('*') if p.is_file()}
    changed = dict(spec, train=dict(spec['train'], **{key: value}))
    stop_path = {
        'config': original['jobs'][0]['config'], 'script': original['jobs'][0]['script'],
        'graph': 'graph.dot', 'plan': 'plan.json',
    }[stop_after]
    real_write = sweep.write_file

    def interrupted_write(path: str, text: str, executable: bool = False):
        real_write(path, text, executable)
        if Path(path) == root / stop_path:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(sweep, 'write_file', interrupted_write)
        with pytest.raises(KeyboardInterrupt):
            _write(changed, force=True)
    published = sweep.load_plan(root)
    assert {p: p.read_bytes() for p in states} == states
    if stop_after == 'plan':  # all files landed before plan.json committed: already safe to run
        assert published['generation'] != original['generation']
        assert not sweep.mismatched_plan_files(published)
        assert sweep.Executor(published, dry_run=True).run() == {'completed': 2}
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        _write(changed)
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files} == before
        return

    assert published == original
    assert sweep.mismatched_plan_files(published)
    with pytest.raises(sweep.SweepError, match='incomplete plan write'):
        sweep.Executor(published, dry_run=True).run()
    # Even the original, logically unchanged spec must repair the partially overwritten files.
    with pytest.raises(sweep.SweepError, match='--force'):
        _write(spec)
    holder = _hold(sweep.plan_lock(str(root)).path)
    try:
        with pytest.raises(sweep.SweepError, match='in use'):
            _write(spec, force=True)
    finally:
        holder.release()
    _, repaired = _write(spec, force=True)
    assert repaired['generation'] != original['generation']
    assert not sweep.mismatched_plan_files(repaired)
    assert {p: p.read_bytes() for p in states} == states
    assert {p for p in root.rglob('*') if p.is_file()} == files  # no additional marker files
    expected = sweep.plan_sweep(spec)
    for job, requested in zip(repaired['jobs'], expected['jobs']):
        assert job['config_hash'] == requested['config_hash']
        assert yaml.safe_load((root / job['config']).read_text()) == requested['_cfg']
    assert sweep.Executor(repaired, dry_run=True).run() == {'completed': 2}


@pytest.mark.usefixtures('restore_parser_defaults')
def test_reporting_changes_need_an_offline_update_but_keep_job_state(tmp_path):
    spec = _spec(tmp_path)
    root, plan = _write(spec)
    _mark_all(root, plan, 'completed')
    states = {p: p.read_bytes() for p in (root / 'state').glob('*.json')}
    changed = dict(spec, metric=dict(name='loss', direction='minimize'))
    with pytest.raises(sweep.SweepError, match='settings changed .*metric'):
        _write(changed)
    holder = _hold(sweep.plan_lock(str(root)).path)
    try:
        with pytest.raises(sweep.SweepError, match='in use'):
            _write(changed, force=True)
    finally:
        holder.release()
    _, updated = _write(changed, force=True)
    assert updated['metric'] == changed['metric']
    assert {p: p.read_bytes() for p in states} == states
    before = (root / 'plan.json').read_bytes()
    _write(changed)
    assert (root / 'plan.json').read_bytes() == before


@pytest.mark.usefixtures('restore_parser_defaults')
@pytest.mark.parametrize('key,value', [
    ('checkpoint_hist', 7), ('log_wandb', True), ('wandb_project', 'sweep-test'), ('wandb_tags', ['sweep', 'test']),
])
def test_output_settings_need_an_offline_update_but_keep_training(
        tmp_path: Path, key: str, value: object,
):
    base = tmp_path / 'base.yaml'
    original = _spec(tmp_path)['train']
    base.write_text(yaml.safe_dump(original))
    spec = _spec(tmp_path, train=dict(config=str(base)))
    root, plan = _write(spec)
    _mark_all(root, plan, 'completed')
    for job in plan['jobs']:
        _fake_run(root / job['dir'], job['end_epoch'])
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}

    # Editing an existing base config must compare with the saved plan, not reread the old spec's base config.
    base.write_text(yaml.safe_dump(dict(original, **{key: value})))
    requested = sweep.plan_sweep(spec)
    changes = sweep.plan_changes(requested)
    assert changes['output_settings'] == [j['id'] for j in plan['jobs']]
    assert not changes['changed']
    with pytest.raises(sweep.SweepError, match='changed output settings'):
        _write(spec)
    holder = _hold(sweep.plan_lock(str(root)).path)
    try:
        with pytest.raises(sweep.SweepError, match='in use'):
            _write(spec, force=True)
    finally:
        holder.release()
    assert {p: p.read_bytes() for p in before} == before

    _, updated = _write(spec, force=True)
    for old, new in zip(plan['jobs'], updated['jobs']):
        assert (old['id'], old['config_hash']) == (new['id'], new['config_hash'])
        assert new['output_settings'][key] == value
        assert yaml.safe_load((root / new['config']).read_text())[key] == value
        state = Path(sweep.JobState(str(root), new).path)
        assert state.read_bytes() == before[state]
        for name in ('summary.csv', 'last.pth.tar'):
            path = root / new['dir'] / name
            assert path.read_bytes() == before[path]
    with pytest.raises(sweep.SweepError, match='replanned'):
        sweep.Executor(plan, dry_run=True).run()

    holder = _hold(sweep.plan_lock(str(root)).path)
    try:
        saved = (root / 'plan.json').stat().st_mtime_ns
        _write(spec)  # an identical request still joins an active sweep without writing
        assert (root / 'plan.json').stat().st_mtime_ns == saved
    finally:
        holder.release()
    base.write_text(yaml.safe_dump(original))  # removing the override also applies the default again
    _, restored = _write(spec, force=True)
    for job in restored['jobs']:
        assert yaml.safe_load((root / job['config']).read_text()) == yaml.safe_load(before[root / job['config']])


@pytest.mark.usefixtures('restore_parser_defaults')
def test_run_spec_checks_the_requested_plan_after_a_concurrent_replan(tmp_path, monkeypatch, caplog):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    path = tmp_path / 'input.yaml'
    path.write_text(yaml.safe_dump(spec))
    write_plan = sweep.write_plan

    def replan_before_worker_starts(plan, requested, force=False):
        root = write_plan(plan, requested, force=force)
        replacement = dict(requested, search=dict(mode='grid', space=dict(lr=[0.9])))
        write_plan(sweep.plan_sweep(replacement), replacement, force=True)
        return root

    monkeypatch.setattr(sweep, 'write_plan', replan_before_worker_starts)
    monkeypatch.setattr(sys, 'argv', ['sweep.py', 'run', str(path), '--dry-run'])
    handler = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(SystemExit) as error:
            sweep.main()
    finally:
        signal.signal(signal.SIGTERM, handler)
    assert error.value.code == 1 and 'replanned' in caplog.text
    assert not list((Path(spec['output']) / spec['name'] / 'state').glob('*.json'))


@pytest.mark.usefixtures('restore_parser_defaults')
def test_runtime_training_overrides_are_rejected(tmp_path, monkeypatch, caplog):
    root, _ = _write(_spec(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['sweep.py', 'run', str(root), '--', '--lr', '0.9'])
    with pytest.raises(SystemExit) as error:
        sweep.main()
    assert error.value.code == 1 and 'Put them in the spec' in caplog.text
    assert not list((root / 'state').glob('*.json'))


@pytest.mark.usefixtures('restore_parser_defaults')
def test_spec_from_cli(tmp_path):
    import argparse
    args = argparse.Namespace(
        name='cli', output=str(tmp_path), sweep=['lr=1e-4,3e-4', 'opt=sgd,adamw'], random=0, seed=1, repeats=1,
        metric=None, gpus_per_job=2, wsd_points=[10], wsd_decay_epochs=[2], wsd_decay_fn=None,
    )
    spec = sweep.spec_from_cli(args, ['--model', 'test_vit', '--warmup-epochs', '2', '--amp'])
    assert spec['search']['space'] == dict(lr=[1e-4, 3e-4], opt=['sgd', 'adamw'])
    assert spec['train'] == dict(model='test_vit', warmup_epochs=2, amp=True)  # only what was given
    assert spec['wsd'] == dict(points=[10], decay_epochs=[2]) and spec['execution'] == dict(gpus_per_job=2)
    plan = sweep.plan_sweep(spec)
    assert plan['jobs'][0]['gpus'] == 2
    sweep.write_plan(plan, spec)
    assert 'nproc_per_node=2' in sweep.run_script_text(plan['jobs'][0], plan)


# -- executor

@pytest.fixture
def image_folder(tmp_path):
    root = tmp_path / 'data'
    for split in ('train', 'validation'):
        for cls, color in (('a', (200, 30, 30)), ('b', (30, 30, 200))):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                Image.new('RGB', (8, 8), color=color).save(d / f'{i}.png')
    return root


@pytest.mark.usefixtures('restore_parser_defaults')
def test_run_wsd_sweep_end_to_end(tmp_path, image_folder, monkeypatch):
    monkeypatch.setenv('PYTHON', sys.executable)
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    spec = _spec(
        tmp_path,
        train=dict(model='test_vit', data_dir=str(image_folder), num_classes=2, device='cpu', workers=0,
                   batch_size=4, warmup_epochs=0, no_aug=True),
        search=dict(mode='grid', space=dict(lr=[0.01])),
        wsd=dict(points=[1, 2], decay_epochs=[1], decay_fn=['linear']),
    )
    plan = sweep.plan_sweep(spec)
    root = sweep.write_plan(plan, spec)
    plan = sweep.load_plan(root)
    counts = sweep.Executor(plan, max_parallel=2, poll_interval=0.5).run()
    assert counts == {'completed': 4}
    trial = Path(root) / plan['trials'][0]['id']
    # one continuous trunk over two segments: a single summary with epochs 0, 1 and both milestones
    with (trial / 'stable' / 'summary.csv').open() as f:
        assert [r['epoch'] for r in csv.DictReader(f)] == ['0', '1']
    assert (trial / 'stable' / 'milestone-1.pth.tar').exists() and (trial / 'stable' / 'milestone-2.pth.tar').exists()
    assert (trial / 'stable' / 'run.log').read_text().count('sweep.py attempt') == 2
    # the cooldown from point 1 decays to zero over one epoch, logged as epoch 1
    with (trial / 'decay-p1-d1-linear' / 'summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [r['epoch'] for r in rows] == ['1'] and float(rows[0]['lr']) == 0.
    results, metric_col = sweep.collect_results(plan)
    assert metric_col == 'eval_top1' and len(results) == 4
    by_job = {r['job']: r for r in results}
    assert by_job['stable-1']['completed_epochs'] == 1 and by_job['stable-2']['completed_epochs'] == 2
    assert by_job['decay-p2-d1-linear']['completed_epochs'] == 3 and by_job['decay-p2-d1-linear']['point'] == 2
    # a second run has nothing to do
    assert sweep.Executor(plan, poll_interval=0.1).run() == {'completed': 4}


@pytest.mark.usefixtures('restore_parser_defaults')
def test_run_marks_failed_and_blocked(tmp_path, image_folder, monkeypatch):
    monkeypatch.setenv('PYTHON', sys.executable)
    spec = _spec(
        tmp_path,
        train=dict(model='not_a_model', data_dir=str(image_folder), num_classes=2, device='cpu', workers=0,
                   batch_size=4, warmup_epochs=0),
        search=dict(mode='grid', space=dict(lr=[0.01])),
        wsd=dict(points=[1], decay_epochs=[1]),
    )
    root, plan = _write(spec)
    executor = sweep.Executor(plan, poll_interval=0.2)
    assert executor.run() == {'failed': 1, 'blocked': 1}
    state = executor.states[plan['jobs'][0]['id']]
    assert state.status == 'failed' and state.attempts[0]['returncode'] != 0
    assert state.attempt_gone() and Path(state.attempt_path).read_text() == ''  # released and retired


def _fake_checkpoint(path, completed_epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': completed_epochs - 1, 'version': 2, 'state_dict': {}}, path)


def _fake_run(job_dir, epochs):
    """The summary and last checkpoint of a run that trained `epochs` epochs."""
    _fake_summary(job_dir / 'summary.csv', epochs)
    _fake_checkpoint(job_dir / 'last.pth.tar', epochs)


def _fake_trunk(stable_dir, epochs, points):
    """A WSD trunk trained `epochs` epochs, with milestones at `points`."""
    _fake_run(stable_dir, epochs)
    for point in points:
        _fake_checkpoint(stable_dir / f'milestone-{point}.pth.tar', point)


def _fake_summary(path, epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'eval_top1'])
        for e in range(epochs):
            w.writerow([e, 1.0, 50.0 + e])


@pytest.mark.usefixtures('restore_parser_defaults')
def test_completion_requires_checkpoints_not_just_summary(tmp_path):
    spec = _spec(tmp_path, train=WSD_TRAIN,
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root, plan = _write(spec)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    executor = sweep.Executor(plan)
    stable = root / trunk['dir']
    # summary says done but the final save never happened (killed / disk full between the two writes)
    _fake_summary(stable / 'summary.csv', 2)
    assert not executor.on_disk_complete(trunk)
    _fake_checkpoint(stable / 'last.pth.tar', 1)  # last is one epoch behind
    assert not executor.on_disk_complete(trunk)
    _fake_checkpoint(stable / 'last.pth.tar', 2)
    assert not executor.on_disk_complete(trunk)  # milestone for the segment's point still missing
    _fake_checkpoint(stable / 'milestone-2.pth.tar', 2)
    assert executor.on_disk_complete(trunk)
    assert not executor.launch(trunk, [])  # settled under the folder locks, nothing left to run
    assert executor.states[trunk['id']].status == 'completed'


@pytest.mark.usefixtures('restore_parser_defaults')
def test_reconcile_leaves_a_job_whose_ranks_hold_its_lock(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    # its worker died (nothing records that), but a rank of the attempt still runs and holds the training lock
    state = sweep.JobState(str(root), job)
    _start_attempt(state, 'a1')
    rank = _join(state, 'a1')
    executor = sweep.Executor(plan)
    executor.reconcile()
    assert executor.states[job['id']].status == 'running'
    assert not executor.launch(job, [])
    # once the rank has exited the attempt is settled, and the job is runnable again
    _exit(rank)
    executor.reconcile()
    assert executor.states[job['id']].status == 'interrupted'
    assert executor.is_ready(job)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_run_sees_state_written_by_another_worker(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    executor = sweep.Executor(plan, poll_interval=0.1)  # loads state: pending
    other = sweep.JobState(root, job)
    other.set('completed')  # another worker finished it meanwhile
    assert executor.run() == {'completed': 1}  # nothing launched (the job would fail, there is no dataset)
    assert not executor.states[job['id']].attempts


@pytest.mark.usefixtures('restore_parser_defaults')
def test_dry_run_with_gpu_pool_walks_the_whole_graph(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])))
    root, plan = _write(spec)
    executor = sweep.Executor(plan, gpus=[0], dry_run=True)
    assert executor.run() == {'completed': 2}
    assert executor.free_gpus == [0]


@pytest.mark.usefixtures('restore_parser_defaults')
def test_total_epochs_matches_scheduler_factory():
    import train
    base = vars(train.parser.parse_args([]))
    assert sweep.total_epochs(dict(base, sched='cosine', epochs=2, lr_cycle_limit=2, cooldown_epochs=1)) == 5
    assert sweep.total_epochs(dict(base, sched='cosine', epochs=10, warmup_epochs=3, warmup_prefix=True)) == 13
    assert sweep.total_epochs(dict(base, sched='step', epochs=10, cooldown_epochs=5)) == 10  # no cooldown for step
    assert sweep.total_epochs(dict(base, sched='wsd', epochs=10, decay_epochs=2, warmup_epochs=1)) == 10
    assert sweep.total_epochs(dict(base, sched='none', epochs=7)) == 7
    with pytest.raises(sweep.SweepError):
        sweep.total_epochs(dict(base, sched='wsd', epochs=10, decay_epochs=20))


@pytest.mark.usefixtures('restore_parser_defaults')
def test_wsd_sched_on_updates_default_yields_to_explicit_values(tmp_path):
    spec = _spec(tmp_path, train=WSD_TRAIN,
                 search=dict(mode='grid', space=dict(sched_on_updates=[False, True])), wsd=dict(points=[3]))
    plan = sweep.plan_sweep(spec)
    assert [t['params']['sched_on_updates'] for t in plan['trials']] == [False, True]
    cfgs = {j['trial']: j['_cfg']['sched_on_updates'] for j in plan['jobs']}
    assert sorted(cfgs.values()) == [False, True]
    base = tmp_path / 'base.yaml'
    base.write_text(yaml.safe_dump(dict(model='test_vit', sched_on_updates=False, warmup_epochs=1)))
    plan = sweep.plan_sweep(_spec(tmp_path, train=dict(config='base.yaml'), wsd=dict(points=[3])), str(tmp_path))
    assert all(j['_cfg']['sched_on_updates'] is False for j in plan['jobs'])
    plan = sweep.plan_sweep(_spec(tmp_path, train=WSD_TRAIN, wsd=dict(points=[3])))
    assert all(j['_cfg']['sched_on_updates'] is True for j in plan['jobs'])


def test_search_space_accepts_yaml_numeric_strings():
    trials = sweep.expand_search(dict(
        mode='random', seed=1, n_trials=3,
        space=dict(lr={'loguniform': ['1e-5', '1e-3']}, wd=['1e-4', '5e-2'], n={'int': ['1', '3']}),
        zip=[dict(a=['1e-1', '2e-1'], b=[1, 2])],
    ))
    assert all(1e-5 <= t['lr'] <= 1e-3 and t['wd'] in (1e-4, 5e-2) and t['n'] in (1, 2, 3) for t in trials)
    assert all(isinstance(t['a'], float) for t in trials)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_default_metric_follows_task(tmp_path):
    plan = sweep.plan_sweep(_spec(tmp_path, train=dict(model='test_vit', task='multilabel', num_classes=2)))
    assert plan['metric'] == {'name': 'map', 'direction': 'maximize'}
    plan = sweep.plan_sweep(_spec(tmp_path, train=dict(model='test_vit', eval_metric='loss')))
    assert plan['metric'] == {'name': 'loss', 'direction': 'minimize'}


def test_aggregate_repeats_handles_list_values():
    rows = [
        dict(job='train', status='completed', input_size=[3, 32, 32], seed=s, eval_top1=v)
        for s, v in ((1, 10.), (2, 20.))
    ] + [dict(job='train', status='completed', input_size=[3, 64, 64], seed=1, eval_top1=30.)]
    agg = sweep.aggregate_repeats(rows, 'eval_top1', ['input_size', 'seed'])
    assert [(r['input_size'], r['n'], r['mean_eval_top1']) for r in agg] == [
        ([3, 32, 32], 2, 15.), ([3, 64, 64], 1, 30.)]


@pytest.mark.usefixtures('restore_parser_defaults')
def test_results_fall_back_to_train_loss_without_validation(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])))
    root, plan = _write(spec)
    for job, losses in zip(plan['jobs'], ((2.0, 1.0), (2.0, 0.5))):
        path = root / job['dir'] / 'summary.csv'
        with path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'train_loss', 'lr'])  # no eval_ columns, no validation set
            for e, loss in enumerate(losses):
                w.writerow([e, loss, 0.1])
    results, metric_col = sweep.collect_results(plan)
    assert metric_col == 'train_loss' and not sweep.metric_maximize(plan, metric_col)
    assert [r['train_loss'] for r in results] == [0.5, 1.0]  # lower loss ranks first
    sweep.show_results(plan)  # the report renders with the fallback column
    assert (root / 'results.csv').exists()



@pytest.mark.usefixtures('restore_parser_defaults')
def test_launch_skips_a_folder_an_attempt_trains_in(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    executor = sweep.Executor(plan)  # selected the job as pending ...
    other = sweep.JobState(root, job)  # ... while another worker started it
    _start_attempt(other, 'a1')
    rank = _join(other, 'a1')
    assert executor.launch(job, []) is False
    assert sweep.JobState(root, job).status == 'running' and len(sweep.JobState(root, job).attempts) == 1
    _exit(rank)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_missing_milestone_is_recovered_from_matching_checkpoint(tmp_path):
    spec = _spec(tmp_path, train=WSD_TRAIN,
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root, plan = _write(spec)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    branch = next(j for j in plan['jobs'] if j['kind'] == 'wsd_cooldown')
    stable = root / trunk['dir']
    _fake_run(stable, 2)  # trained to the stop epoch, the milestone save failed
    executor = sweep.Executor(plan)
    assert not executor.launch(trunk, [])  # settled instead
    assert (stable / 'milestone-2.pth.tar').exists()
    assert executor.states[trunk['id']].status == 'completed' and executor.is_ready(branch)
    # last has moved on (a later segment) but the ranked checkpoint for that epoch is there
    (stable / 'milestone-2.pth.tar').unlink()
    _fake_checkpoint(stable / 'last.pth.tar', 3)
    _fake_checkpoint(stable / 'checkpoint-1.pth.tar', 2)
    executor = sweep.Executor(plan)
    executor.recover_artifacts(trunk)
    assert (stable / 'milestone-2.pth.tar').exists() and sweep.checkpoint_epochs(stable / 'milestone-2.pth.tar') == 2
    # nothing at that epoch: left missing, so the job isn't complete
    (stable / 'milestone-2.pth.tar').unlink()
    (stable / 'checkpoint-1.pth.tar').unlink()
    executor.recover_artifacts(trunk)
    assert not (stable / 'milestone-2.pth.tar').exists() and not executor.on_disk_complete(trunk)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_idle_worker_waits_for_jobs_running_elsewhere(tmp_path):
    import threading
    spec = _spec(tmp_path, train=WSD_TRAIN,
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root, plan = _write(spec)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    branch = next(j for j in plan['jobs'] if j['kind'] == 'wsd_cooldown')
    # another worker runs the trunk, its rank holds the training lock
    other = sweep.JobState(str(root), trunk)
    _start_attempt(other, 'a1')
    rank = _join(other, 'a1')
    executor = sweep.Executor(plan, poll_interval=0.1)
    launched = []

    def fake_launch(job, gpus):  # don't really run the branch, pretend it finished
        launched.append(job['id'])
        executor.states[job['id']].set('completed')
        return False

    executor.launch = fake_launch
    result = {}
    worker = threading.Thread(target=lambda: result.update(counts=executor.run()), daemon=True)
    worker.start()
    time.sleep(0.5)
    assert worker.is_alive() and not launched  # waiting, not exited
    # the other worker finishes the trunk
    stable = root / trunk['dir']
    _fake_run(stable, 2)
    _fake_checkpoint(stable / 'milestone-2.pth.tar', 2)
    other.set('completed')
    _exit(rank)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert launched == [branch['id']] and result['counts'] == {'completed': 2}


@pytest.mark.usefixtures('restore_parser_defaults')
def test_total_epochs_accepts_fractional_update_decay():
    import train
    base = vars(train.parser.parse_args([]))
    cfg = dict(base, sched='wsd', epochs=2, warmup_epochs=0, decay_epochs=0.2, sched_on_updates=True)
    assert sweep.total_epochs(cfg) == 2
    with pytest.raises(sweep.SweepError):  # still impossible: decay longer than the schedule
        sweep.total_epochs(dict(cfg, decay_epochs=5))



@pytest.mark.usefixtures('restore_parser_defaults')
def test_waiting_worker_notices_another_worker_dying(tmp_path):
    import threading
    spec = _spec(tmp_path, train=WSD_TRAIN,
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root, plan = _write(spec)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    branch = next(j for j in plan['jobs'] if j['kind'] == 'wsd_cooldown')
    other = sweep.JobState(str(root), trunk)
    _start_attempt(other, 'a1')
    rank = _join(other, 'a1')  # alive for now
    executor = sweep.Executor(plan, poll_interval=0.1)
    launched = []

    def fake_launch(job, gpus):
        launched.append(job['id'])
        if job is trunk:  # pretend the trunk trained to completion
            stable = root / trunk['dir']
            _fake_run(stable, 2)
            _fake_checkpoint(stable / 'milestone-2.pth.tar', 2)
        executor.states[job['id']].set('completed')
        return False

    executor.launch = fake_launch
    worker = threading.Thread(target=executor.run, daemon=True)
    worker.start()
    time.sleep(0.4)
    assert worker.is_alive() and not launched
    # the other worker and its rank die without updating the state file, their locks go with them
    _exit(rank)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert launched == [trunk['id'], branch['id']]


@pytest.mark.usefixtures('restore_parser_defaults', 'isolate_cli_backend_flags')
def test_cli_run_exits_nonzero_on_failure(tmp_path, image_folder, monkeypatch):
    monkeypatch.setenv('PYTHON', sys.executable)
    monkeypatch.setattr('sys.argv', [
        'sweep.py', 'run', '--name', 'fail', '--output', str(tmp_path), '--sweep', 'lr=0.1', '--poll', '0.2', '--quiet',
        '--', '--model', 'not_a_model', '--data-dir', str(image_folder), '--num-classes', '2', '--device', 'cpu',
        '--epochs', '1',
    ])
    with pytest.raises(SystemExit) as e:
        sweep.main()
    assert e.value.code == 1
    # a dry run of the same sweep is not a failure
    monkeypatch.setattr('sys.argv', ['sweep.py', 'run', str(tmp_path / 'fail'), '--dry-run'])
    sweep.main()


@pytest.mark.usefixtures('restore_parser_defaults')
def test_reconcile_leaves_locally_running_jobs_to_finish(tmp_path):
    spec = _spec(tmp_path, train=WSD_TRAIN,
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root, plan = _write(spec)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    executor = sweep.Executor(plan)
    state = executor.states[trunk['id']]
    state.start_attempt('a1', [], ['run.sh'])
    train = sweep.Lock(state.train_path)
    assert train.acquire(exclusive=False)
    executor.running[trunk['id']] = (None, [], None, train)  # this worker is running it
    # the final checkpoint has landed but the process is still finishing up (milestone not written yet)
    stable = root / trunk['dir']
    _fake_run(stable, 2)
    executor.reconcile()
    assert state.status == 'running'
    assert not (stable / 'milestone-2.pth.tar').exists()  # no recovery under a live process either
    train.release()


# -- executor safety and WSD trunk extension


@pytest.mark.usefixtures('restore_parser_defaults')
def test_executor_rejects_jobs_larger_than_the_gpu_pool(tmp_path):
    spec = _spec(tmp_path, execution=dict(gpus_per_job=2))
    root, plan = _write(spec)
    with pytest.raises(sweep.SweepError, match='pool has 1'):
        sweep.Executor(plan, gpus=[0])
    sweep.Executor(plan, gpus=[0, 1])
    sweep.Executor(plan)  # no pool, jobs see every GPU


@pytest.mark.usefixtures('restore_parser_defaults')
def test_adding_a_wsd_point_extends_the_trunk(tmp_path):
    spec = _spec(tmp_path, train=WSD_TRAIN, search=dict(mode='grid', space=dict(lr=[0.1])),
                 wsd=dict(points=[2, 3], decay_epochs=[1]))
    root, plan = _write(spec)
    stable = root / next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')['dir']
    _fake_trunk(stable, 3, (2, 3))
    _mark_all(root, plan, 'completed')
    extended = dict(spec, wsd=dict(points=[2, 3, 5], decay_epochs=[1]))
    with pytest.raises(sweep.SweepError, match='2 jobs added'):  # the stable-5 segment and its cooldown
        _write(extended)
    _, plan = _write(extended, force=True)
    assert (stable / 'milestone-3.pth.tar').exists() and (stable / 'last.pth.tar').exists()
    executor = sweep.Executor(plan, dry_run=True)
    states = {j['id'].split('/', 1)[1]: executor.states[j['id']].status for j in plan['jobs']}
    assert states == {
        'stable-2': 'completed', 'decay-p2-d1-1-sqrt': 'completed', 'stable-3': 'completed',
        'decay-p3-d1-1-sqrt': 'completed', 'stable-5': 'pending', 'decay-p5-d1-1-sqrt': 'pending',
    }
    trunk_job = next(j for j in plan['jobs'] if j['id'].endswith('stable-5'))
    trunk_cfg = yaml.safe_load((root / trunk_job['config']).read_text())
    assert trunk_cfg['save_epochs'] == [2, 3, 5] and trunk_cfg['stop_epoch'] == 5
    assert [j['id'].split('/', 1)[1] for j in executor.ready_jobs()] == ['stable-5']


@pytest.mark.usefixtures('restore_parser_defaults')
def test_sweep_checkpoint_history_default_and_overrides(tmp_path):
    plan = sweep.plan_sweep(_spec(tmp_path))
    assert {j['_cfg']['checkpoint_hist'] for j in plan['jobs']} == {1}
    plan = sweep.plan_sweep(_spec(tmp_path, train=dict(model='test_vit', checkpoint_hist=10)))
    assert {j['_cfg']['checkpoint_hist'] for j in plan['jobs']} == {10}
    # given on the command line at train.py's default, it is still an explicit choice
    args = argparse.Namespace(
        sweep=['lr=0.1'], random=0, seed=42, repeats=1, name='cli', output=str(tmp_path), gpus_per_job=1,
        metric=None, wsd_points=None, wsd_decay_epochs=None, wsd_decay_fn=None)
    spec = sweep.spec_from_cli(args, ['--model', 'test_vit', '--checkpoint-hist', '10'])
    assert spec['train']['checkpoint_hist'] == 10
    spec = sweep.spec_from_cli(args, ['--model', 'test_vit', '--checkpoint-hist=10'])
    assert spec['train']['checkpoint_hist'] == 10
    assert 'checkpoint_hist' not in sweep.spec_from_cli(args, ['--model', 'test_vit'])['train']


@pytest.mark.usefixtures('restore_parser_defaults')
def test_ready_jobs_put_trunk_segments_first(tmp_path):
    spec = _spec(tmp_path, train=WSD_TRAIN, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])),
                 wsd=dict(points=[2, 4], decay_epochs=[1]))
    plan = sweep.plan_sweep(spec)
    executor = sweep.Executor(plan, dry_run=True)
    executor.states[plan['jobs'][0]['id']].data['status'] = 'completed'  # trial 0 stable-2
    assert [j['kind'] for j in executor.ready_jobs()] == ['wsd_stable', 'wsd_stable', 'wsd_cooldown']


@pytest.mark.usefixtures('restore_parser_defaults')
def test_milestone_recovery_waits_for_the_summary(tmp_path, monkeypatch):
    spec = _spec(tmp_path, train=WSD_TRAIN, search=dict(mode='grid', space=dict(lr=[0.1])),
                 wsd=dict(points=[2, 4], decay_epochs=[1]))
    root, plan = _write(spec)
    stable = root / plan['jobs'][0]['dir']
    _fake_run(stable, 2)
    _fake_checkpoint(stable / 'milestone-2.pth.tar', 2)
    loads = []
    real = sweep.checkpoint_epochs
    monkeypatch.setattr(sweep, 'checkpoint_epochs', lambda path: loads.append(path) or real(path))
    executor = sweep.Executor(plan)
    executor.recover_artifacts(next(j for j in plan['jobs'] if j['id'].endswith('stable-4')))
    assert not loads  # the trunk is at 2 epochs, nothing can be at the 4 epoch milestone yet


@pytest.mark.usefixtures('restore_parser_defaults')
def test_failed_attempts_by_other_workers_count_toward_retries(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    other = sweep.JobState(root, job)
    other.start_attempt('a1', [], ['run.sh'])
    other.attempts[-1]['start'] = '2099-01-01T00:00:00+00:00'  # a fast clock on the other host
    other.set('failed', returncode=1)
    executor = sweep.Executor(plan)
    assert executor.is_ready(job)  # a new run retries it
    other.reload()
    other.start_attempt('a2', [], ['run.sh'])  # another worker retried it since this one started, and it failed again
    other.attempts[-1]['start'] = '2000-01-01T00:00:00+00:00'  # its clock went backwards; it still counts
    other.set('failed', returncode=1)
    executor.refresh_states()
    assert not executor.is_ready(job)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_sweep_timestamps_are_utc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as patch:
        patch.setenv('TZ', 'America/Vancouver')
        time.tzset()
        try:
            root, plan = _write(_spec(tmp_path))
            executor = sweep.Executor(plan)
            job = plan['jobs'][0]
            state = executor.states[job['id']]
            _start_attempt(state, 'a1', returncode=1)
            executor.reconcile()
            assert state.status == 'failed'
            for stamp in (plan['created'], state.attempts[-1]['start'], state.attempts[-1]['end']):
                assert datetime.fromisoformat(stamp).utcoffset() == timedelta(0)
        finally:
            patch.undo()
            time.tzset()


@pytest.mark.usefixtures('restore_parser_defaults')
def test_launch_rechecks_the_retry_budget_under_the_locks(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    executor = sweep.Executor(plan, retries=0)
    assert executor.is_ready(job)
    other = sweep.JobState(root, job)  # another worker runs and fails it after this one selected it
    other.start_attempt('a1', [], ['run.sh'])
    other.set('failed', returncode=1)
    assert not executor.launch(job, [])
    state = sweep.JobState(root, job)
    assert len(state.attempts) == 1 and state.status == 'failed'
    assert state.attempt_gone()  # its locks were released


@pytest.mark.usefixtures('restore_parser_defaults')
def test_replan_publishes_files_atomically_and_unchanged_plan_writes_nothing(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    config, script = root / job['config'], root / job['script']
    inodes = config.stat().st_ino, script.stat().st_ino
    _write(spec)
    assert (config.stat().st_ino, script.stat().st_ino) == inodes  # unchanged, untouched
    os.link(config, root / 'config.yaml.old')  # a link sees an in-place rewrite, but not a replacement
    changed = _spec(tmp_path, train=dict(spec['train'], weight_decay=0.1), search=spec['search'])
    _write(changed, force=True)
    assert 'weight_decay: 0.1' not in (root / 'config.yaml.old').read_text()
    assert 'weight_decay: 0.1' in config.read_text() and os.access(script, os.X_OK)
    assert not list(root.rglob('*.tmp'))


@pytest.mark.usefixtures('restore_parser_defaults')
def test_gpu_ids_index_into_an_inherited_cuda_visible_devices(tmp_path, monkeypatch):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    _, plan = _write(spec)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert sweep.Executor(plan, gpus=[0, 1]).visible_devices([1]) == '1'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,6')
    executor = sweep.Executor(plan, gpus=[0, 1])
    assert executor.visible_devices([1]) == '6' and executor.visible_devices([0, 1]) == '4,6'
    with pytest.raises(sweep.SweepError, match='outside the inherited'):
        sweep.Executor(plan, gpus=[0, 2])


@pytest.mark.usefixtures('restore_parser_defaults')
def test_cooldown_table_compares_trials_at_equal_budgets(tmp_path, capsys):
    spec = _spec(tmp_path, train=WSD_TRAIN, search=dict(mode='grid', space=dict(lr=[0.1, 0.2]), repeats=2),
                 wsd=dict(points=[2, 3], decay_epochs=[1]))
    root, plan = _write(spec)
    trials = {t['id']: t for t in plan['trials']}
    for job in plan['jobs']:
        trial = trials[job['trial']]
        if job['id'].endswith('decay-p3-d1-1-sqrt') and trial['params']['lr'] == 0.1 and trial['repeat'] == 1:
            continue  # still to run
        base = 50. + (5. if trial['params']['lr'] == 0.2 else 0.) + trial['repeat']
        _fake_summary(root / job['dir'] / 'summary.csv', job['end_epoch'])
        rows = (root / job['dir'] / 'summary.csv').read_text().splitlines()
        rows = rows[:1] + [f'{e},1.0,{base + e}' for e in range(job['end_epoch'])]
        (root / job['dir'] / 'summary.csv').write_text('\n'.join(rows) + '\n')
        for path in job['artifacts']:
            _fake_checkpoint(root / path, job['end_epoch'])
        sweep.JobState(str(root), job).set('completed')
    results, metric_col = sweep.collect_results(plan)
    rows, columns = sweep.cooldown_table(plan, results, metric_col)
    assert columns == ['trial', 'lr', 'p2-d1', 'p3-d1']  # repeats are averaged, the seed is not a column
    by_lr = {r['lr']: r for r in rows}
    assert [r['lr'] for r in rows] == [0.2, 0.1]  # ordered by the longest budget
    assert by_lr[0.2]['p2-d1'] == '57.5000*' and by_lr[0.1]['p2-d1'] == '52.5000'  # mean of repeats 0 and 1
    assert by_lr[0.1]['p3-d1'] == '53.0000 (1/2)'
    sweep.show_results(plan)
    out = capsys.readouterr().out
    assert 'stable' not in out and 'p3-d1' in out  # trunk segments are left to results.csv
    with (root / 'results.csv').open() as f:
        written = list(csv.DictReader(f))
    assert len(written) == len(results) and {'budget', 'completed_epochs', 'decay_fn'} <= set(written[0])


@pytest.mark.usefixtures('restore_parser_defaults')
def test_plain_results_rank_within_each_budget(tmp_path, capsys):
    spec = _spec(tmp_path, train=WSD_TRAIN,
                 search=dict(mode='grid', space=dict(lr=[0.1, 0.2], epochs=[2, 4])))
    root, plan = _write(spec)
    for job in plan['jobs']:
        _fake_run(root / job['dir'], job['end_epoch'])
    results, _ = sweep.collect_results(plan)
    assert [r['budget'] for r in results] == [2, 2, 4, 4]
    sweep.show_results(plan)
    out = capsys.readouterr().out
    assert '2 epoch budget' in out and '4 epoch budget' in out


@pytest.mark.usefixtures('restore_parser_defaults')
def test_diverged_runs_rank_last(tmp_path, capsys):
    """A nan metric (diverged run) ranks last and is never the best value, instead of scrambling the ranking."""
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2, 0.4]), repeats=2))
    root, plan = _write(spec)
    trials = {t['id']: t for t in plan['trials']}
    final = {0.1: '60.0', 0.2: 'nan', 0.4: '70.0'}
    for job in plan['jobs']:
        path = root / job['dir'] / 'summary.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        lr = trials[job['trial']]['params']['lr']
        path.write_text(f'epoch,train_loss,eval_top1\n0,1.0,55.0\n1,1.0,{final[lr]}\n')
    _mark_all(root, plan, 'completed')
    results, metric_col = sweep.collect_results(plan)
    assert [r['lr'] for r in results] == [0.4, 0.4, 0.1, 0.1, 0.2, 0.2]
    assert [r[f'best_{metric_col}'] for r in results if r['lr'] == 0.2] == [55.0, 55.0]
    sweep.show_results(plan)
    agg = capsys.readouterr().out.split('Over 2 repeats:')[1].strip().splitlines()[1:4]
    assert [line.split()[0] for line in agg] == ['0.4', '0.1', '0.2']


@pytest.mark.usefixtures('restore_parser_defaults')
def test_unchanged_wsd_plan_with_lr_noise_replans_cleanly(tmp_path):
    spec = _spec(tmp_path, train=dict(WSD_TRAIN, lr_noise=[0.5, 1.0]), search=dict(mode='grid', space=dict(lr=[0.1])),
                 wsd=dict(points=[2, 4], decay_epochs=[1]))
    _write(spec)
    _write(spec)  # the segments of one trunk must not conflict with each other


# -- coordination through flock: the plan lock, a job folder's locks and its live attempt

def _hold(path, exclusive=False):
    lock = sweep.Lock(str(path))
    assert lock.acquire(exclusive=exclusive)
    return lock


def _held_elsewhere(path):
    """Whether another process is kept from taking the lock exclusively."""
    code = (f'import fcntl, os\nfd = os.open({str(path)!r}, os.O_RDWR)\n'
            f'try:\n    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept BlockingIOError:\n    print("held")')
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.strip() == 'held'


@pytest.mark.usefixtures('restore_parser_defaults')
def test_a_started_sweep_can_only_be_extended(tmp_path):
    spec = _spec(tmp_path, train=WSD_TRAIN, search=dict(mode='grid', space=dict(lr=[0.1])),
                 wsd=dict(points=[2, 4], decay_epochs=[1]))
    root, plan = _write(spec)
    changed = dict(spec, train=dict(WSD_TRAIN, weight_decay=0.1))
    _write(changed, force=True)  # nothing has started, any change goes
    _, plan = _write(spec, force=True)
    sweep.JobState(str(root), plan['jobs'][0]).start_attempt('a1', [], ['run.sh'])  # recorded before it starts
    for edit, match in (
            (changed, 'changed'),
            (dict(spec, wsd=dict(points=[2, 3, 4], decay_epochs=[1])), 'changed .* after the existing ones'),
            (dict(spec, wsd=dict(points=[4], decay_epochs=[1])), 'removed'),
            (dict(spec, search=dict(mode='grid', space=dict(lr=[0.05, 0.1]))), 'removed'),  # shifts the trials
    ):
        before = (root / 'plan.json').read_text()
        with pytest.raises(sweep.SweepError, match=f'only be extended .*{match}'):
            _write(edit, force=True)
        assert (root / 'plan.json').read_text() == before
    added = dict(spec, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])),
                 wsd=dict(points=[2, 4, 6], decay_epochs=[1, 2]))
    with pytest.raises(sweep.SweepError, match='--force'):
        _write(added)
    _, plan = _write(added, force=True)
    assert len(plan['jobs']) == 2 * (3 + 3 * 2)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_planning_waits_for_every_worker_and_rank_to_stop(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    changed = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])))
    holder = _hold(sweep.plan_lock(str(root)).path)  # a worker (even an idle one) or a rank of one of its jobs
    _write(spec)  # an unchanged plan is fine while the sweep runs
    with pytest.raises(sweep.SweepError, match='--force'):  # a missing --force is reported first
        _write(changed)
    with pytest.raises(sweep.SweepError, match='in use'):
        _write(changed, force=True)
    assert len(sweep.load_plan(root)['jobs']) == 1
    holder.release()
    _write(changed, force=True)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_a_worker_waits_for_a_planner_and_runs_the_plan_it_loaded(tmp_path):
    import threading
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    planner = _hold(sweep.plan_lock(str(root)).path, exclusive=True)
    executor = sweep.Executor(plan, poll_interval=0.05)
    launched = []

    def fake_launch(job, gpus):
        launched.append(job['id'])
        executor.states[job['id']].set('completed')
        return False

    executor.launch = fake_launch
    result = {}
    worker = threading.Thread(target=lambda: result.update(counts=executor.run()), daemon=True)
    worker.start()
    time.sleep(0.3)
    assert worker.is_alive() and not launched
    planner.release()
    worker.join(timeout=5)
    assert result['counts'] == {'completed': 1}
    # a worker whose plan was replaced before it got going refuses to run the old one
    stale = sweep.Executor(plan)
    _write(_spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2]))), force=True)
    with pytest.raises(sweep.SweepError, match='replanned'):
        stale.run()
    assert not _held_elsewhere(sweep.plan_lock(str(root)).path)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_ranks_join_only_the_live_attempt(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    state = sweep.JobState(str(root), plan['jobs'][0])
    _start_attempt(state, 'a1')
    rank = _join(state, 'a1')
    assert not state.attempt_gone()  # the rank holds the job's training lock ...
    with pytest.raises(sweep.SweepError, match='in use'):  # ... and the plan lock
        _write(_spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2]))), force=True)
    with pytest.raises(SystemExit, match='not the live one'):
        _join(state, 'a0')
    _exit(rank)
    assert state.attempt_gone()
    state.set_live_attempt('')
    with pytest.raises(SystemExit, match='retired'):
        _join(state, 'a1')


@pytest.mark.usefixtures('restore_parser_defaults')
def test_coordination_arguments_stay_out_of_the_training_arguments(tmp_path, monkeypatch):
    import train
    locks = [tmp_path / 'plan.lock', tmp_path / 'job.train']
    for path in locks:
        path.touch()
    (tmp_path / 'job.attempt').write_text('a1')
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--model', 'test_vit', '--sweep-locks', *map(str, locks),
        '--sweep-attempt', str(tmp_path / 'job.attempt'), 'a1'])
    try:
        args, args_text = train._parse_args()
        assert all(_held_elsewhere(path) for path in locks)
    finally:
        _exit(train._sweep_lock_fds)
        train._sweep_lock_fds.clear()
    assert args.model == 'test_vit' and not any('sweep' in k for k in vars(args)) and 'sweep' not in args_text
    assert not any('sweep' in k for k in sweep.train_defaults())  # nor in a sweep's job configs or their hashes


@pytest.mark.usefixtures('restore_parser_defaults')
def test_finish_settles_only_once_every_rank_has_exited(tmp_path):
    import threading
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    executor = sweep.Executor(plan, gpus=[0])
    state = executor.states[job['id']]
    # an attempt launched as Executor.launch does, its torchrun is killed and leaves a rank running
    coord, train = executor.lock_folder(job)
    state.start_attempt('a1', [0], ['run.sh'])
    state.set_live_attempt('a1')
    train.share()
    coord.release()
    rank = _join(state, 'a1')
    reader = threading.Thread(target=lambda: None)
    reader.start()
    executor.running[job['id']] = (None, [0], reader, train)
    executor.free_gpus.clear()
    executor.finish(job['id'], -9)
    state.reload()
    assert state.status == 'running' and state.attempts[-1]['returncode'] == -9 and not state.attempt_gone()
    assert not executor.free_gpus and len(executor.draining) == executor.max_parallel
    with pytest.raises(SystemExit, match='retired'):  # no rank of it joins any more
        _join(state, 'a1')
    executor.reconcile()
    assert executor.states[job['id']].status == 'running'
    assert not executor.free_gpus
    _exit(rank)
    executor.reconcile()
    assert executor.states[job['id']].status == 'failed'  # with the exit code recorded by finish
    assert executor.free_gpus == [0] and not executor.draining


@pytest.mark.usefixtures('restore_parser_defaults')
def test_draining_attempt_keeps_its_parallel_slot_without_a_gpu_pool(tmp_path):
    import threading
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])))
    root, plan = _write(spec)
    first, second = plan['jobs']
    executor = sweep.Executor(plan, max_parallel=1, poll_interval=0.1)
    # this worker's torchrun of the first job exited, a rank of it still runs
    state = executor.states[first['id']]
    _start_attempt(state, 'a1', returncode=-9)
    rank = _join(state, 'a1')
    state.set_live_attempt('')
    executor.draining[first['id']] = ('a1', [])
    launched = []

    def fake_launch(job, gpus):
        launched.append(job['id'])
        executor.states[job['id']].set('completed')
        return False

    executor.launch = fake_launch
    worker = threading.Thread(target=executor.run, daemon=True)
    worker.start()
    try:
        time.sleep(0.5)
        assert worker.is_alive() and not launched  # the draining attempt holds the only slot
    finally:
        _exit(rank)
    worker.join(timeout=5)
    assert not worker.is_alive() and second['id'] in launched and not executor.draining

@pytest.mark.usefixtures('restore_parser_defaults')
def test_draining_attempt_does_not_touch_another_workers_retry(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    executor = sweep.Executor(plan, gpus=[0])
    state = executor.states[job['id']]
    _start_attempt(state, 'a1', returncode=-9)
    state.set_live_attempt('')
    executor.free_gpus.clear()
    executor.draining[job['id']] = ('a1', [0])
    # Another worker settles the old attempt and starts its retry before this one polls again.
    other = sweep.Executor(plan)
    other.reconcile()
    replacement = other.states[job['id']]
    _start_attempt(replacement, 'a2')
    rank = _join(replacement, 'a2')
    try:
        before = Path(state.path).read_bytes()
        executor.refresh_states()
        executor.reconcile()
        assert executor.free_gpus == [0] and not executor.draining
        assert Path(state.path).read_bytes() == before
        assert Path(state.attempt_path).read_text() == 'a2'
        executor.reconcile()
        assert executor.free_gpus == [0]  # released once
    finally:
        _exit(rank)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_a_late_rank_cannot_join_an_attempt_retired_without_replacement(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    state = sweep.JobState(str(root), job)
    # the worker started the attempt and died before any rank joined it
    _start_attempt(state, 'a1')
    executor = sweep.Executor(plan, retries=0)
    executor.reconcile()
    assert executor.states[job['id']].status == 'interrupted'
    with pytest.raises(SystemExit, match='retired'):
        _join(state, 'a1')
    # nor once it is settled as failed, with no retries left
    state.reload()
    _start_attempt(state, 'a2', returncode=1)
    executor.refresh_states()
    executor.reconcile()
    assert executor.states[job['id']].status == 'failed' and not executor.is_ready(job)
    with pytest.raises(SystemExit, match='retired'):
        _join(state, 'a2')


@pytest.mark.usefixtures('restore_parser_defaults')
def test_a_probe_never_disturbs_a_held_lock(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    state = sweep.JobState(str(root), plan['jobs'][0])
    for exclusive in (False, True):
        held = _hold(state.train_path, exclusive)
        assert not state.attempt_gone() and not state.attempt_gone()
        assert _held_elsewhere(state.train_path)
        with pytest.raises(AssertionError):  # a held lock is never upgraded (or re-taken) in place
            held.acquire(exclusive=True)
        held.release()
        assert state.attempt_gone()


@pytest.mark.usefixtures('restore_parser_defaults')
def test_status_and_results_change_nothing(tmp_path, capsys):
    spec = _spec(tmp_path, train=WSD_TRAIN, search=dict(mode='grid', space=dict(lr=[0.1])),
                 wsd=dict(points=[2], decay_epochs=[1]))
    root, plan = _write(spec)
    trunk = plan['jobs'][0]
    _fake_run(root / trunk['dir'], 2)  # complete but for the milestone, which a worker would recover
    sweep.JobState(str(root), trunk).start_attempt('a1', [], ['run.sh'])  # its worker died
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file() and 'locks' not in p.parts}
    sweep.show_status(plan)
    assert 'running (gone)' in capsys.readouterr().out
    sweep.show_results(plan, csv_path=str(tmp_path / 'results.csv'))
    after = {p: p.read_bytes() for p in root.rglob('*') if p.is_file() and 'locks' not in p.parts}
    assert after == before


_STUB = r"""
import os, sys, time
sys.stdout = sys.stderr = open(os.devnull, 'w')  # its worker's pipe may be gone
sys.path.insert(0, {timm_root!r})
time.sleep(float(os.environ.get('STUB_DELAY', '0')))  # a rank slow to start
import torch
import train
args, _ = train.config_parser.parse_known_args(sys.argv[1:])
try:
    train._join_sweep_attempt(args.sweep_locks, *args.sweep_attempt)
except SystemExit:
    with open(os.environ['STUB_LOG'], 'a') as f:
        f.write('rejected\n')
    raise
rank = int(os.environ.get('RANK', '0'))
distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
if distributed:
    torch.distributed.init_process_group('gloo')
    model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    with open(os.environ['STUB_LOG'] + '.' + str(rank) + '.pid', 'w') as f:
        f.write(str(os.getpid()))
with open(os.environ['STUB_LOG'], 'a') as f:
    f.write('joined\n')
deadline = time.time() + 120  # never outlive a failed test for long
while not os.path.exists(os.environ['STUB_GO']) and time.time() < deadline:
    time.sleep(0.05)
out = {out!r}
if rank == 0:
    with open(os.path.join(out, 'summary.csv'), 'w') as f:
        f.write('epoch,train_loss,eval_top1\n' + ''.join(f'{{e}},1.0,{{50 + e}}\n' for e in range({epochs})))
    torch.save({{'epoch': {epochs} - 1, 'version': 2}}, os.path.join(out, 'last.pth.tar'))
if distributed:
    torch.distributed.destroy_process_group()
"""


def _wait_for(condition, timeout=60.):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.05)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f'/proc/{pid}/stat') as f:
        return f.read().split(')')[-1].split()[0] != 'Z'


@pytest.fixture
def two_rank_sweep(tmp_path, monkeypatch):
    if sys.platform != 'linux':
        pytest.skip('these process-lifecycle checks use Linux /proc and loopback naming')
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])), execution=dict(gpus_per_job=2))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    stub = tmp_path / 'ranks.py'
    stub.write_text(_STUB.format(timm_root=sweep.TIMM_ROOT, out=str(root / job['dir']), epochs=job['end_epoch']))
    (root / job['script']).write_text(
        f'#!/bin/bash\n# plan_generation: {plan["generation"]}\n'
        f'exec "{sys.executable}" -m torch.distributed.run --standalone --nproc-per-node=2 '
        f'"{stub}" "$@"\n')
    monkeypatch.setenv('STUB_LOG', str(tmp_path / 'ranks.log'))
    monkeypatch.setenv('STUB_GO', str(tmp_path / 'go'))
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('GLOO_SOCKET_IFNAME', 'lo')
    yield root, plan
    (tmp_path / 'go').touch()
    for path in tmp_path.glob('ranks.log.*.pid'):
        _wait_for(lambda: not _alive(int(path.read_text())))


@pytest.mark.usefixtures('restore_parser_defaults')
def test_killed_torchrun_keeps_gpus_reserved_until_both_ranks_exit(two_rank_sweep, tmp_path, monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    root, plan = two_rank_sweep
    job = plan['jobs'][0]
    executor = sweep.Executor(plan, gpus=[0, 1], echo=False)
    executor.free_gpus.clear()
    assert executor.launch(job, [0, 1])
    proc = executor.running[job['id']][0]
    try:
        _wait_for(lambda: proc.poll() is not None or len(list(tmp_path.glob('ranks.log.*.pid'))) == 2)
        assert proc.poll() is None, (root / job['dir'] / 'run.log').read_text()
        proc.kill()
        proc.wait(timeout=10)
        executor.poll()
        state = executor.states[job['id']]
        assert state.status == 'running' and not executor.free_gpus
        assert job['id'] in executor.draining and not state.attempt_gone()
        # Only the ranks hold the plan lock here: this executor was launched directly, outside run().
        lock = sweep.plan_lock(str(root))
        assert not lock.acquire(exclusive=True)
        assert not sweep.Executor(plan).launch(job, [])
        (tmp_path / 'go').touch()
        _wait_for(state.attempt_gone)
        executor.reconcile()
        assert not executor.draining and executor.free_gpus == [0, 1]
        assert state.status == 'completed'  # valid outputs win even though the launcher was killed
        assert lock.acquire(exclusive=True)
        lock.release()
    finally:
        (tmp_path / 'go').touch()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)


@pytest.mark.usefixtures('restore_parser_defaults')
@pytest.mark.parametrize('stop_signal', [signal.SIGKILL, signal.SIGTERM])
def test_two_rank_job_after_worker_stops(two_rank_sweep, tmp_path, stop_signal):
    root, plan = two_rank_sweep
    job = plan['jobs'][0]
    state = sweep.JobState(str(root), job)
    command = [sys.executable, os.path.join(sweep.TIMM_ROOT, 'sweep.py'), 'run', str(root), '--poll', '0.1', '--quiet']
    with (tmp_path / 'worker.log').open('w') as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        _wait_for(lambda: worker.poll() is not None or len(list(tmp_path.glob('ranks.log.*.pid'))) == 2)
        assert worker.poll() is None, (tmp_path / 'worker.log').read_text()
        worker.send_signal(stop_signal)
        worker.wait(timeout=40)
        state.reload()
        if stop_signal == signal.SIGKILL:
            assert state.status == 'running' and not state.attempt_gone()
            lock = sweep.plan_lock(str(root))
            assert not lock.acquire(exclusive=True)
            assert not sweep.Executor(plan).launch(job, [])
            (tmp_path / 'go').touch()
        else:
            assert state.status == 'interrupted' and Path(state.attempt_path).read_text() == ''
        _wait_for(state.attempt_gone)
        executor = sweep.Executor(plan)
        executor.reconcile()
        assert executor.states[job['id']].status == ('completed' if stop_signal == signal.SIGKILL else 'interrupted')
        lock = sweep.plan_lock(str(root))
        assert lock.acquire(exclusive=True)
        lock.release()
    finally:
        (tmp_path / 'go').touch()
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=40)


@pytest.mark.usefixtures('restore_parser_defaults')
@pytest.mark.skipif(sys.platform != 'linux', reason='process-lifecycle checks use Linux /proc')
def test_a_job_outlives_its_killed_worker_and_late_ranks_stay_out(tmp_path, monkeypatch):
    """Real processes: a worker is SIGKILLed while its job runs, then while its job is slow to start."""
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root, plan = _write(spec)
    job = plan['jobs'][0]
    stub = tmp_path / 'stub.py'
    stub.write_text(_STUB.format(timm_root=sweep.TIMM_ROOT, out=str(root / job['dir']), epochs=job['end_epoch']))
    (root / job['script']).write_text(
        f'#!/bin/bash\n# plan_generation: {plan["generation"]}\nexec "{sys.executable}" "{stub}" "$@"\n')
    joined, go = tmp_path / 'ranks.log', tmp_path / 'go'
    monkeypatch.setenv('STUB_LOG', str(joined))
    monkeypatch.setenv('STUB_GO', str(go))
    state = sweep.JobState(str(root), job)
    worker_cmd = [sys.executable, os.path.join(sweep.TIMM_ROOT, 'sweep.py'), 'run', str(root), '--poll', '0.1',
                  '--quiet']

    def start_worker():
        state.reload()
        started = len(state.attempts)
        worker = subprocess.Popen(worker_cmd, cwd=sweep.TIMM_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_for(lambda: state.reload() or len(state.attempts) > started and 'pid' in state.attempts[-1])
        return worker, state.attempts[-1]['pid']

    try:
        _outlive_a_killed_worker(root, job, state, start_worker, joined, go, monkeypatch, tmp_path)
    finally:
        go.touch()  # releases any stub still waiting


def _outlive_a_killed_worker(root, job, state, start_worker, joined, go, monkeypatch, tmp_path):
    # 1. the worker dies while its job trains: nothing replans or restarts the job until the job exits
    worker, pid = start_worker()
    _wait_for(joined.exists)
    worker.kill()
    worker.wait()
    with pytest.raises(sweep.SweepError, match='in use'):
        _write(_spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2]))), force=True)
    other = sweep.Executor(sweep.load_plan(root))
    other.reconcile()
    assert not other.launch(job, []) and other.states[job['id']].status == 'running'
    go.touch()
    _wait_for(lambda: not _alive(pid))
    other.reconcile()
    assert other.states[job['id']].status == 'completed'

    # 2. the worker dies before its job's rank joins: the attempt is retired, the late rank exits untrained
    for name in ('summary.csv', 'last.pth.tar'):
        (root / job['dir'] / name).unlink()
    state.reload()
    state.set('pending')
    go.unlink()
    joined.unlink()
    monkeypatch.setenv('STUB_DELAY', '4')
    worker, pid = start_worker()
    worker.kill()
    worker.wait()
    other.refresh_states()
    other.reconcile()
    assert other.states[job['id']].status == 'interrupted'
    _wait_for(lambda: not _alive(pid))
    assert joined.read_text() == 'rejected\n'
    monkeypatch.setenv('STUB_DELAY', '0')
    go.touch()
    assert sweep.Executor(sweep.load_plan(root), poll_interval=0.1, echo=False).run() == {'completed': 1}
    assert joined.read_text() == 'rejected\njoined\n'  # only the replacement attempt trained
