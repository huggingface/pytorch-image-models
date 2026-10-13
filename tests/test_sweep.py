"""Tests for sweep.py: search space expansion, plan generation, WSD branching and the local executor."""
import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

import sweep

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
    root = sweep.write_plan(sweep.plan_sweep(spec), spec)
    assert (Path(root) / 'plan.json').exists() and (Path(root) / 'graph.dot').exists()
    job = sweep.load_plan(root)['jobs'][0]
    assert (Path(root) / job['config']).exists() and os.access(Path(root) / job['script'], os.X_OK)
    sweep.write_plan(sweep.plan_sweep(spec), spec)  # same plan, fine
    changed = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.3])))
    with pytest.raises(sweep.SweepError, match='different'):
        sweep.write_plan(sweep.plan_sweep(changed), changed)
    sweep.write_plan(sweep.plan_sweep(changed), changed, force=True)
    assert len(sweep.load_plan(root)['jobs']) == 2


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
    assert by_job['stable-1']['epochs'] == 1 and by_job['stable-2']['epochs'] == 2
    assert by_job['decay-p2-d1-linear']['epochs'] == 3 and by_job['decay-p2-d1-linear']['point'] == 2
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
    root = sweep.write_plan(sweep.plan_sweep(spec), spec)
    plan = sweep.load_plan(root)
    executor = sweep.Executor(plan, poll_interval=0.2)
    assert executor.run() == {'failed': 1, 'blocked': 1}
    state = executor.states[plan['jobs'][0]['id']]
    assert state.status == 'failed' and state.attempts[0]['returncode'] != 0
    assert not os.path.exists(state.claim_dir)


# -- review fixes

def _fake_checkpoint(path, completed_epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': completed_epochs - 1, 'version': 2, 'state_dict': {}}, path)


def _fake_summary(path, epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'eval_top1'])
        for e in range(epochs):
            w.writerow([e, 1.0, 50.0 + e])


@pytest.mark.usefixtures('restore_parser_defaults')
def test_completion_requires_checkpoints_not_just_summary(tmp_path):
    spec = _spec(tmp_path, train=dict(model='test_vit', warmup_epochs=1),
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec))
    plan = sweep.load_plan(root)
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
    executor.reconcile()
    assert executor.states[trunk['id']].status == 'completed'


@pytest.mark.usefixtures('restore_parser_defaults')
def test_reconcile_respects_live_training_child(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root = sweep.write_plan(sweep.plan_sweep(spec), spec)
    plan = sweep.load_plan(root)
    job = plan['jobs'][0]
    # orchestrator died (no claim dir) but the training process it recorded is alive: leave the job alone
    state = sweep.JobState(root, job)
    state.start_attempt(os.getpid(), [], ['run.sh'])
    executor = sweep.Executor(plan)
    executor.reconcile()
    assert executor.states[job['id']].status == 'running'
    # with a dead training pid it is interrupted and runnable again
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    state.attempts[-1]['pid'] = dead.pid
    state.save()
    executor = sweep.Executor(plan)
    executor.reconcile()
    assert executor.states[job['id']].status == 'interrupted'
    assert executor.is_ready(job)


@pytest.mark.usefixtures('restore_parser_defaults')
def test_run_sees_state_written_by_another_worker(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root = sweep.write_plan(sweep.plan_sweep(spec), spec)
    plan = sweep.load_plan(root)
    job = plan['jobs'][0]
    executor = sweep.Executor(plan, poll_interval=0.1)  # loads state: pending
    other = sweep.JobState(root, job)
    other.set('completed')  # another worker finished it meanwhile
    assert executor.run() == {'completed': 1}  # nothing launched (the job would fail, there is no dataset)
    assert not executor.launched_attempts


@pytest.mark.usefixtures('restore_parser_defaults')
def test_dry_run_with_gpu_pool_walks_the_whole_graph(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])))
    root = sweep.write_plan(sweep.plan_sweep(spec), spec)
    plan = sweep.load_plan(root)
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
    spec = _spec(tmp_path, train=dict(model='test_vit', warmup_epochs=1),
                 search=dict(mode='grid', space=dict(sched_on_updates=[False, True])), wsd=dict(points=[3]))
    plan = sweep.plan_sweep(spec)
    assert [t['params']['sched_on_updates'] for t in plan['trials']] == [False, True]
    cfgs = {j['trial']: j['_cfg']['sched_on_updates'] for j in plan['jobs']}
    assert sorted(cfgs.values()) == [False, True]
    base = tmp_path / 'base.yaml'
    base.write_text(yaml.safe_dump(dict(model='test_vit', sched_on_updates=False, warmup_epochs=1)))
    plan = sweep.plan_sweep(_spec(tmp_path, train=dict(config='base.yaml'), wsd=dict(points=[3])), str(tmp_path))
    assert all(j['_cfg']['sched_on_updates'] is False for j in plan['jobs'])
    plan = sweep.plan_sweep(_spec(tmp_path, train=dict(model='test_vit', warmup_epochs=1), wsd=dict(points=[3])))
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
def test_force_replan_resets_changed_jobs_and_sets_outputs_aside(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec))
    job = sweep.load_plan(root)['jobs'][0]
    old_hash = job['config_hash']
    _fake_summary(root / job['dir'] / 'summary.csv', 10)
    _fake_checkpoint(root / job['dir'] / 'last.pth.tar', 10)
    sweep.JobState(str(root), job).set('completed')
    # an unswept parameter changes: same job id, different config
    changed = _spec(tmp_path, train=dict(model='test_vit', data_dir='/data', num_classes=2, warmup_epochs=1, epochs=10,
                                         weight_decay=0.1), search=dict(mode='grid', space=dict(lr=[0.1])))
    sweep.write_plan(sweep.plan_sweep(changed), changed, force=True)
    plan = sweep.load_plan(root)
    assert plan['jobs'][0]['config_hash'] != old_hash
    assert not (root / job['dir'] / 'summary.csv').exists()
    assert (root / f'{job["dir"]}.stale-{old_hash}' / 'summary.csv').exists()
    executor = sweep.Executor(plan)
    executor.reconcile()
    assert executor.states[job['id']].status == 'pending' and executor.is_ready(job)
    # an unchanged plan written with force leaves everything alone
    sweep.write_plan(sweep.plan_sweep(changed), changed, force=True)
    assert (root / f'{job["dir"]}.stale-{old_hash}').exists()


@pytest.mark.usefixtures('restore_parser_defaults')
def test_results_fall_back_to_train_loss_without_validation(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec))
    plan = sweep.load_plan(root)
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


# -- second review round

@pytest.mark.usefixtures('restore_parser_defaults')
def test_reintroduced_trial_with_new_config_does_not_reuse_old_outputs(tmp_path):
    base_train = dict(model='test_vit', data_dir='/data', num_classes=2, warmup_epochs=1, epochs=10)
    spec_a = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1, 0.2])))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec_a), spec_a))
    job = sweep.load_plan(root)['jobs'][1]  # trial-001-lr0.2
    old_hash = job['config_hash']
    _fake_summary(root / job['dir'] / 'summary.csv', 10)
    _fake_checkpoint(root / job['dir'] / 'last.pth.tar', 10)
    sweep.JobState(str(root), job).set('completed')
    # drop the trial, then bring it back with an unswept parameter changed
    spec_b = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    sweep.write_plan(sweep.plan_sweep(spec_b), spec_b, force=True)
    spec_c = _spec(tmp_path, train=dict(base_train, weight_decay=0.1),
                   search=dict(mode='grid', space=dict(lr=[0.1, 0.2])))
    sweep.write_plan(sweep.plan_sweep(spec_c), spec_c, force=True)
    plan = sweep.load_plan(root)
    new_job = plan['jobs'][1]
    assert new_job['id'] == job['id'] and new_job['config_hash'] != old_hash
    assert not (root / job['dir'] / 'summary.csv').exists()
    assert (root / f'{job["dir"]}.stale-{old_hash}' / 'summary.csv').exists()
    assert sweep.recorded_config_hash(root / new_job['config']) == new_job['config_hash']
    executor = sweep.Executor(plan)
    executor.reconcile()
    assert executor.states[job['id']].status == 'pending'


@pytest.mark.usefixtures('restore_parser_defaults')
def test_claim_recheck_leaves_running_job_alone(tmp_path):
    spec = _spec(tmp_path, search=dict(mode='grid', space=dict(lr=[0.1])))
    root = sweep.write_plan(sweep.plan_sweep(spec), spec)
    plan = sweep.load_plan(root)
    job = plan['jobs'][0]
    executor = sweep.Executor(plan)
    # another worker is between releasing its claim and ... no: it has published 'running' and the claim is free
    # for a moment (e.g. it is restarting the attempt); we must not turn that into 'completed'
    other = sweep.JobState(root, job)
    other.start_attempt(os.getpid(), [], ['run.sh'])
    other.release()
    assert executor.launch(job, []) is False
    assert sweep.JobState(root, job).status == 'running'
    assert not os.path.exists(other.claim_dir)  # our claim was released again


@pytest.mark.usefixtures('restore_parser_defaults')
def test_missing_milestone_is_recovered_from_matching_checkpoint(tmp_path):
    spec = _spec(tmp_path, train=dict(model='test_vit', warmup_epochs=1),
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec))
    plan = sweep.load_plan(root)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    branch = next(j for j in plan['jobs'] if j['kind'] == 'wsd_cooldown')
    stable = root / trunk['dir']
    _fake_summary(stable / 'summary.csv', 2)
    _fake_checkpoint(stable / 'last.pth.tar', 2)  # trained to the stop epoch, the milestone save failed
    executor = sweep.Executor(plan)
    executor.reconcile()
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
    spec = _spec(tmp_path, train=dict(model='test_vit', warmup_epochs=1),
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec))
    plan = sweep.load_plan(root)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    branch = next(j for j in plan['jobs'] if j['kind'] == 'wsd_cooldown')
    # another worker holds the trunk: it is 'running' with a live pid and a claim
    other = sweep.JobState(str(root), trunk)
    assert other.claim()
    other.start_attempt(os.getpid(), [], ['run.sh'])
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
    _fake_summary(stable / 'summary.csv', 2)
    _fake_checkpoint(stable / 'last.pth.tar', 2)
    _fake_checkpoint(stable / 'milestone-2.pth.tar', 2)
    other.set('completed')
    other.release()
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


# -- third review round

@pytest.mark.usefixtures('restore_parser_defaults')
def test_force_replan_resets_every_segment_sharing_the_stable_dir(tmp_path):
    train = dict(model='test_vit', data_dir='/data', num_classes=2, warmup_epochs=0)
    spec = _spec(tmp_path, train=train, search=dict(mode='grid', space=dict(lr=[0.1])),
                 wsd=dict(points=[1, 2], decay_epochs=[1]))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec))
    plan = sweep.load_plan(root)
    segments = [j for j in plan['jobs'] if j['kind'] == 'wsd_stable']
    stable = root / segments[0]['dir']
    _fake_summary(stable / 'summary.csv', 2)
    _fake_checkpoint(stable / 'last.pth.tar', 2)
    for point in (1, 2):
        _fake_checkpoint(stable / f'milestone-{point}.pth.tar', point)
    for job in plan['jobs']:
        sweep.JobState(str(root), job).set('completed')
    # an unswept parameter changes: both trunk segments share the moved directory and must both be reset
    changed = _spec(tmp_path, train=dict(train, weight_decay=0.1), search=dict(mode='grid', space=dict(lr=[0.1])),
                    wsd=dict(points=[1, 2], decay_epochs=[1]))
    sweep.write_plan(sweep.plan_sweep(changed), changed, force=True)
    plan = sweep.load_plan(root)
    executor = sweep.Executor(plan)
    executor.reconcile()
    assert {executor.states[j['id']].status for j in plan['jobs']} == {'pending'}
    assert not (stable / 'summary.csv').exists()
    assert len(list(root.glob(f'{segments[0]["dir"]}.stale-*'))) == 1


@pytest.mark.usefixtures('restore_parser_defaults')
def test_waiting_worker_notices_an_owner_dying(tmp_path):
    import threading
    spec = _spec(tmp_path, train=dict(model='test_vit', warmup_epochs=1),
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec))
    plan = sweep.load_plan(root)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    branch = next(j for j in plan['jobs'] if j['kind'] == 'wsd_cooldown')
    other = sweep.JobState(str(root), trunk)
    assert other.claim()
    other.start_attempt(os.getpid(), [], ['run.sh'])  # alive for now
    executor = sweep.Executor(plan, poll_interval=0.1)
    launched = []

    def fake_launch(job, gpus):
        launched.append(job['id'])
        if job is trunk:  # pretend the trunk trained to completion
            stable = root / trunk['dir']
            _fake_summary(stable / 'summary.csv', 2)
            _fake_checkpoint(stable / 'last.pth.tar', 2)
            _fake_checkpoint(stable / 'milestone-2.pth.tar', 2)
        executor.states[job['id']].set('completed')
        return False

    executor.launch = fake_launch
    worker = threading.Thread(target=executor.run, daemon=True)
    worker.start()
    time.sleep(0.4)
    assert worker.is_alive() and not launched
    # the other worker and its training child die without updating the state file
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    other.reload()
    other.attempts[-1]['pid'] = dead.pid
    other.save()
    with open(os.path.join(other.claim_dir, 'owner'), 'w') as f:
        f.write(f'{socket.gethostname()}:{dead.pid}\n')
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
    spec = _spec(tmp_path, train=dict(model='test_vit', warmup_epochs=1),
                 search=dict(mode='grid', space=dict(lr=[0.1])), wsd=dict(points=[2], decay_epochs=[1]))
    root = Path(sweep.write_plan(sweep.plan_sweep(spec), spec))
    plan = sweep.load_plan(root)
    trunk = next(j for j in plan['jobs'] if j['kind'] == 'wsd_stable')
    executor = sweep.Executor(plan)
    state = executor.states[trunk['id']]
    assert state.claim()
    state.start_attempt(os.getpid(), [], ['run.sh'])
    executor.running[trunk['id']] = (None, [], None, None)  # this worker is running it
    # the final checkpoint has landed but the process is still finishing up (milestone not written yet)
    stable = root / trunk['dir']
    _fake_summary(stable / 'summary.csv', 2)
    _fake_checkpoint(stable / 'last.pth.tar', 2)
    executor.reconcile()
    assert state.status == 'running' and os.path.exists(state.claim_dir)
    assert not (stable / 'milestone-2.pth.tar').exists()  # no recovery under a live process either
