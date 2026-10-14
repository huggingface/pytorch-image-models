"""Randomized sequences against sweep.py's planner and its flock coordination of workers and job ranks.

The first test replans a sweep offline under random spec edits while training its jobs in between (fully, partially,
or crashing before a milestone). Simulated checkpoints record the config identity they were trained under, and
cooldowns the identity of the trunk milestone they branched from. Checked:

- an unchanged plan is accepted without --force and changes nothing, any other change needs --force
- once the sweep has started a plan is only accepted if it adds jobs, a refused plan changes nothing
- a job only resumes state trained under its own config, a cooldown only branches from its own trunk
- the plan can always be trained to completion

The second test runs workers (Executors) in one process against simulated job processes, whose ranks take the locks
as train.py does. Workers die (losing every lock, possibly in the middle of launching or finishing a job), are
stopped, and come back. Ranks start late, crash, outlive their torchrun, and join in the gaps where a worker briefly
holds no lock. Planners try to extend the sweep at random moments. Checked:

- at most one attempt runs in a job folder, and only an attempt the job's record names as running
- a rank only joins an attempt that is still live, an attempt is never settled while one of its ranks runs
- a plan change is accepted exactly when no worker and no joined rank of the sweep is alive
- once the chaos stops, the sweep runs to completion and every lock is free

Raise the number of sequences for a soak run, e.g. SWEEP_FUZZ_SEEDS=2000 pytest -n 8 tests/test_sweep_fuzz.py, and
set SWEEP_FUZZ_DIR to also run them on another filesystem, e.g. an NFS mount.
"""
import copy
import csv
import io
import logging
import os
import random
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

import sweep

fcntl = pytest.importorskip('fcntl')
pytestmark = pytest.mark.skipif(not sweep.TORCH_SUPPORTED, reason=sweep.TORCH_REQUIREMENT)

SEEDS = int(os.environ.get('SWEEP_FUZZ_SEEDS', 24))
FUZZ_DIR = os.environ.get('SWEEP_FUZZ_DIR')


class Violation(AssertionError):
    pass


@pytest.fixture
def work_dir(tmp_path):
    if not FUZZ_DIR:
        yield tmp_path
        return
    path = Path(tempfile.mkdtemp(dir=FUZZ_DIR, prefix='sweep-fuzz-'))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def quiet():
    logging.disable(logging.WARNING)  # interrupted / failed / recovered jobs are logged by the hundred
    yield
    logging.disable(logging.NOTSET)


def _load(path):
    return torch.load(path, weights_only=True)


def _save(path, epochs, ident, src=None):
    state = {'epoch': epochs - 1, 'version': 2, 'ident': ident}
    if src is not None:
        state['src'] = src
    torch.save(state, path)


def _write_summary(job_dir, start, end):
    """Summary rows for epochs [start, end), keeping earlier rows as a resumed train.py does."""
    path = job_dir / 'summary.csv'
    rows = []
    if path.exists() and start > 0:
        with path.open() as f:
            rows = [r for r in list(csv.reader(f))[1:] if int(r[0]) < start]
    rows += [[e, 1.0, 50.0 + e] for e in range(start, end)]
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'eval_top1'])
        writer.writerows(rows)


def _resume_epochs(last, ident, job_id):
    """Epochs in the job's own last checkpoint, which must have been trained under `ident`."""
    if not last.exists():
        return 0
    state = _load(last)
    if state['ident'] != ident:
        raise Violation(f'{job_id} resumes state trained as {state["ident"]}, its identity is {ident}')
    return state['epoch'] + 1


def _train_outputs(root, plan, job, how='complete'):
    """What train.py leaves in a job's folder, checking it never consumes stale state. `how` is complete, partial
    (one epoch, then it dies) or no_milestone (it dies between the last checkpoint and the milestone). True if the
    run completed.
    """
    job_dir = root / job['dir']
    job_dir.mkdir(parents=True, exist_ok=True)
    last = job_dir / 'last.pth.tar'
    src = None
    if job['kind'] == 'wsd_cooldown':
        src = _load(root / job['inputs'][0])['ident']
        trunk = next(j for j in plan['jobs'] if j['id'] == job['deps'][0])
        if src != trunk['config_hash']:
            raise Violation(f'{job["id"]} branches from a milestone trained as {src}, trunk is {trunk["config_hash"]}')
        start = _resume_epochs(last, job['config_hash'], job['id']) or job['wsd']['point']
        if last.exists() and _load(last)['src'] != src:
            raise Violation(f'{job["id"]} resumes a partial run branched from another trunk')
    else:
        start = _resume_epochs(last, job['config_hash'], job['id'])
    end = job['wsd']['point'] if job['kind'] == 'wsd_stable' else job['end_epoch']
    if start > end:
        raise Violation(f'{job["id"]} launched with {start} epochs already trained, it stops at {end}')
    if how == 'partial' and end - start > 1:
        _write_summary(job_dir, start, start + 1)
        _save(last, start + 1, job['config_hash'], src)
        return False
    _write_summary(job_dir, start, end)
    _save(last, end, job['config_hash'], src)
    if job['kind'] == 'wsd_stable':
        if how == 'no_milestone':
            return False
        shutil.copy(last, job_dir / f'milestone-{end}.pth.tar')
    return True


def _check_completed(root, plan):
    jobs = {j['id']: j for j in plan['jobs']}
    for job in plan['jobs']:
        if sweep.JobState(str(root), job).status != 'completed':
            continue
        if job['kind'] == 'wsd_stable':
            milestone = root / job['artifacts'][1]
            if not milestone.exists() or _load(milestone)['ident'] != job['config_hash']:
                raise Violation(f'{job["id"]} completed without a milestone trained under its config')
            continue
        state = _load(root / job['dir'] / 'last.pth.tar')
        if state['ident'] != job['config_hash'] or state['epoch'] + 1 < job['end_epoch']:
            raise Violation(f'{job["id"]} completed with state {state["ident"]} at {state["epoch"] + 1} epochs')
        if job['kind'] == 'wsd_cooldown' and state['src'] != jobs[job['deps'][0]]['config_hash']:
            raise Violation(f'{job["id"]} completed from another trunk')


def _random_spec(tmp, rng, name='fuzz'):
    train = dict(model='test_vit', data_dir='/data', num_classes=2, warmup_epochs=1,
                 weight_decay=rng.choice([0.05, 0.1]))
    if rng.random() < 0.2:
        train['lr_noise'] = [0.5, 1.0]  # ties a trunk to its horizon, a later point then changes it
    search = dict(mode='grid', space=dict(lr=rng.choice([[0.1], [0.1, 0.2]])))
    if rng.random() < 0.2:
        search['repeats'] = 2
    spec = dict(name=name, output=str(tmp), train=train, search=search)
    if rng.random() < 0.25:
        train['epochs'] = rng.choice([3, 4])  # a plain sweep
        return spec
    spec['wsd'] = dict(
        points=sorted(rng.sample([2, 3, 4, 5, 6, 7, 8], rng.randint(1, 3))),
        decay_epochs=rng.choice([[1], [1, 2], [2]]),
        decay_fn=rng.choice([['1-sqrt'], ['linear'], ['1-sqrt', 'linear']]),
    )
    return spec


def _edit(spec, rng, additive=False):
    """A random spec edit, `additive` ones only add jobs (unless lr noise ties trunks to their horizon)."""
    kinds = ['add_trial'] + (['append_point', 'append_point', 'add_decay'] if 'wsd' in spec else [])
    if not additive:
        kinds += ['spec', 'config', 'prepend_trial'] + (['insert_point', 'drop_point'] if 'wsd' in spec else [])
    kind = rng.choice(kinds)
    if kind == 'spec':
        return _random_spec(spec['output'], rng, spec['name'])
    spec = copy.deepcopy(spec)
    lrs = spec['search']['space']['lr']
    wsd = spec.get('wsd') or {}
    points = wsd.get('points')
    if kind == 'config':
        spec['train']['weight_decay'] = 0.15 if spec['train']['weight_decay'] != 0.15 else 0.05
    elif kind == 'add_trial':
        lrs.append(round(max(lrs) * 2, 6))
    elif kind == 'prepend_trial':
        lrs.insert(0, round(min(lrs) / 2, 6))
    elif kind == 'append_point':
        points.append(points[-1] + rng.randint(1, 2))
    elif kind == 'add_decay':
        wsd['decay_epochs'] = wsd['decay_epochs'] + [max(wsd['decay_epochs']) + 1]
    elif kind == 'insert_point':
        free = [p for p in range(2, points[-1]) if p not in points]
        if free:
            points.insert(0, rng.choice(free))
            points.sort()
    elif kind == 'drop_point' and len(points) > 1:
        points.remove(rng.choice(points))
    return spec


def _write_spec(spec, force=False):
    return Path(sweep.write_plan(sweep.plan_sweep(spec), spec, force=force))


def _snapshot(root):
    """Every file of the sweep except the lock files, with its content."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*'))
        if p.is_file() and sweep.LOCK_DIR not in str(p.relative_to(root))
    }


def _kind_of_change(old_plan, new_plan):
    """'same', 'additive' (only adds jobs) or 'other', judged on what identifies each job."""
    old = {j['id']: j for j in old_plan['jobs']}
    new = {j['id']: j for j in new_plan['jobs']}
    keys = ('config_hash', 'kind', 'dir', 'deps', 'inputs', 'end_epoch')
    if set(old) - set(new) or any(tuple(j[k] for k in keys) != tuple(new[i][k] for k in keys) for i, j in old.items()):
        return 'other'
    return 'same' if set(new) == set(old) else 'additive'


# -- offline replans

def _train_some(root, plan, rng, budget):
    """Settle and train up to `budget` ready jobs one at a time, as a single worker would."""
    executor = sweep.Executor(plan, retries=1000)
    for _ in range(budget):
        executor.refresh_states()
        ready = executor.ready_jobs()
        if not ready:
            return executor
        job = rng.choice(ready[:3])
        coord, train = executor.lock_folder(job)
        executor.settle(job)  # recovers a missing milestone, as a launch would
        coord.release()
        train.release()
        state = executor.states[job['id']]
        if not executor.is_ready(job):
            continue
        how = rng.choices(['complete', 'partial', 'no_milestone'], [6, 3, 1])[0]
        state.start_attempt(f'a{len(state.attempts)}', [], ['run.sh'])
        if _train_outputs(root, plan, job, how):
            state.set('completed')
        else:
            state.set('failed', returncode=1)
    return executor


@pytest.mark.parametrize('seed', range(SEEDS))
def test_offline_replans(work_dir, quiet, seed):
    rng = random.Random(seed)
    spec = _random_spec(work_dir, rng)
    root = _write_spec(spec)
    for _ in range(14):
        plan = sweep.load_plan(root)
        if rng.random() < 0.5:
            _train_some(root, plan, rng, rng.randint(1, 6))
            _check_completed(root, sweep.load_plan(root))
            continue
        new_spec = _edit(spec, rng)
        force = rng.random() < 0.7
        change = _kind_of_change(plan, sweep.plan_sweep(new_spec))
        started = any((root / 'state').glob('*.json'))
        refuse = change != 'same' and (not force or (started and change == 'other'))
        before = _snapshot(root)
        try:
            _write_spec(new_spec, force=force)
        except sweep.SweepError as e:
            if not refuse:
                raise Violation(f'{change} change (force={force}, started={started}) refused: {e}')
            if _snapshot(root) != before:
                raise Violation(f'a refused plan changed the sweep: {e}')
            continue
        if refuse:
            raise Violation(f'{change} change (force={force}, started={started}) was accepted')
        if change == 'same' and _snapshot(root) != before:
            raise Violation('an unchanged plan changed the sweep')
        spec = new_spec
        _check_completed(root, sweep.load_plan(root))
    plan = sweep.load_plan(root)
    executor = _train_some(root, plan, rng, 1000)
    left = [j['id'] for j in plan['jobs'] if executor.states[j['id']].status != 'completed']
    assert not left, f'nothing ready, yet not complete: {left}'
    _check_completed(root, plan)


# -- concurrent workers and ranks

class Died(Exception):
    """A worker killed in the middle of what it was doing."""


class Rank:
    def __init__(self, attempt):
        self.attempt = attempt
        self.state = 'starting'  # -> joined | rejected | exited
        self.late = attempt.sim.rng.random() < 0.3  # slow to start, e.g. still importing when its worker dies
        self.fds = []

    def try_join(self):
        """train.py's _join_sweep_attempt, with its blocking locks emulated by retries: 'waiting' if a lock is held
        exclusively right now.
        """
        fds = []
        for path in self.attempt.lock_paths:
            fd = os.open(path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                for held in fds:
                    os.close(held)
                return 'waiting'
            fds.append(fd)
        with open(self.attempt.attempt_path) as f:
            live = f.read().strip()
        if live != self.attempt.id:
            for fd in fds:
                os.close(fd)
            self.state = 'rejected'
            return self.state
        self.fds, self.state = fds, 'joined'
        self.attempt.sim.check_joined(self)
        return self.state

    def exit(self):
        for fd in self.fds:
            os.close(fd)
        self.fds = []
        self.state = 'exited'


class FakeProc:
    """A job's torchrun agent and its ranks, created where the executor would start run.sh."""

    def __init__(self, sim, command):
        self.sim = sim
        self.lock_paths = command[command.index('--sweep-locks') + 1:command.index('--sweep-attempt')]
        self.attempt_path, self.id = command[command.index('--sweep-attempt') + 1:][:2]
        self.job_id = next(j['id'] for j in sim.plan()['jobs'] if command[0].endswith(j['script']))
        self.ranks = [Rank(self) for _ in range(sim.rng.choice([1, 2]))]
        self.returncode = None  # the agent's
        self.pid = 4_000_000 + len(sim.procs)
        self.stdout = io.StringIO('')
        sim.procs.append(self)
        sim.chaos()  # started, but the worker may die before it records the process

    def poll(self):
        return self.returncode

    def terminate(self):  # the agent forwards SIGTERM to its ranks
        if self.returncode is None:
            for rank in self.ranks:
                rank.exit()
            self.returncode = -15

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode

    def agent_alive(self):
        return self.returncode is None

    def rank_failed(self):
        """A rank exited early, a live agent tears the others down and fails."""
        if self.agent_alive():
            for rank in self.ranks:
                rank.exit()
            self.returncode = 1

    def train_to_end(self, how):
        """Every rank has joined, rank 0 trains and they all exit."""
        job = next(j for j in self.sim.plan()['jobs'] if j['id'] == self.job_id)
        completed = _train_outputs(self.sim.root, self.sim.plan(), job, how)
        for rank in self.ranks:
            rank.exit()
        if self.agent_alive():
            self.returncode = 0 if completed else 1


class Sim:
    def __init__(self, root, rng):
        self.root = root
        self.rng = rng
        self.procs = []
        self.workers = []  # (executor, its plan lock, the locks it ever took)
        self.current = None  # the worker acting right now, chaos may kill it
        self._plan = None

    def plan(self):
        if self._plan is None:
            self._plan = sweep.load_plan(self.root)
        return self._plan

    def live_ranks(self):
        return [r for p in self.procs for r in p.ranks if r.state == 'joined']

    def check_joined(self, rank):
        state = sweep.JobState(str(self.root), self.job(rank.attempt.job_id))
        last = state.attempts[-1] if state.attempts else {}
        if state.status != 'running' or last.get('id') != rank.attempt.id:
            raise Violation(f'a rank joined attempt {rank.attempt.id} of {rank.attempt.job_id}, its record has '
                            f'{state.status} attempt {last.get("id")}')

    def job(self, job_id):
        return next(j for j in self.plan()['jobs'] if j['id'] == job_id)

    def check(self):
        folders = {}
        for rank in self.live_ranks():
            self.check_joined(rank)  # never settled, nor replaced, under a joined rank
            folder = self.job(rank.attempt.job_id)['dir']
            folders.setdefault(folder, set()).add(rank.attempt.id)
        for folder, attempts in folders.items():
            if len(attempts) > 1:
                raise Violation(f'attempts {sorted(attempts)} run in {folder} at once')

    # -- workers

    def start_worker(self):
        executor = sweep.Executor(self.plan(), retries=1000, poll_interval=0., echo=False)
        lock = sweep.Lock(sweep.plan_lock(str(self.root)).path)
        if not lock.acquire(exclusive=False):
            raise Violation('a worker could not take the plan lock shared with no planner running')
        self.workers.append((executor, lock, []))

    def kill_worker(self, worker):
        """Its process dies: every descriptor closes, its jobs' processes carry on."""
        executor, lock, taken = worker
        for held in taken + [lock]:
            held.release()
        self.workers.remove(worker)

    def stop_worker(self, worker):
        """Ctrl-C or SIGTERM: its jobs are stopped and settled, then it exits."""
        self.act(worker, worker[0].terminate)
        if worker in self.workers:
            self.kill_worker(worker)

    def act(self, worker, fn, *args):
        self.current = worker
        try:
            fn(*args)
        except Died:
            self.current = None
            self.kill_worker(worker)
        finally:
            self.current = None

    def step_worker(self, worker):
        executor = worker[0]

        def step():
            executor.poll()
            executor.refresh_states()
            executor.reconcile()
            for job in executor.ready_jobs():
                if len(executor.running) + len(executor.draining) >= 3:
                    break
                executor.launch(job, [])
        self.act(worker, step)

    def chaos(self):
        """Called whenever a worker writes an attempt or drops a lock: a starting rank may try to join in the gap,
        and the acting worker may die right there.
        """
        if self.rng.random() < 0.3:
            starting = [r for p in self.procs for r in p.ranks if r.state == 'starting' and not r.late]
            if starting:
                rank = self.rng.choice(starting)
                if rank.try_join() == 'rejected':
                    rank.attempt.rank_failed()
        if self.current is not None and self.rng.random() < 0.03:
            raise Died()

    # -- planners

    def try_replan(self, spec):
        busy = bool(self.workers) or bool(self.live_ranks())
        new_spec = _edit(spec, self.rng, additive=True)
        additive = _kind_of_change(self.plan(), sweep.plan_sweep(new_spec)) == 'additive'
        try:
            _write_spec(new_spec, force=True)
        except sweep.SweepError as e:
            if busy and 'in use' in str(e) or not additive and 'only be extended' in str(e):
                return spec
            raise Violation(f'planning refused (busy={busy}): {e}')
        if busy:
            raise Violation('the plan changed while workers or ranks of the sweep were alive')
        self._plan = None
        return new_spec


def _patch_hooks(monkeypatch, sim):
    real_acquire, real_release = sweep.Lock.acquire, sweep.Lock.release
    real_set_live = sweep.JobState.set_live_attempt

    def acquire(self, exclusive, block=False):
        ok = real_acquire(self, exclusive, block)
        if ok and sim.current is not None:
            sim.current[2].append(self)  # closed if the worker dies
        return ok

    def release(self):
        held = self.fd is not None
        real_release(self)
        if held and sim.current is not None:
            sim.chaos()

    def share(self):  # flock's own conversion: unlock, then lock shared, a waiting rank may get in between
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        sim.chaos()
        fcntl.flock(self.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)

    def set_live_attempt(self, attempt_id=''):
        real_set_live(self, attempt_id)
        if sim.current is not None:
            sim.chaos()

    monkeypatch.setattr(sweep.Lock, 'acquire', acquire)
    monkeypatch.setattr(sweep.Lock, 'release', release)
    monkeypatch.setattr(sweep.Lock, 'share', share)
    monkeypatch.setattr(sweep.JobState, 'set_live_attempt', set_live_attempt)
    monkeypatch.setattr(sweep.subprocess, 'Popen', lambda command, **kwargs: FakeProc(sim, command))


def _rank_step(sim):
    """One thing happens to the job processes: a rank starts, a job trains to its end, a rank crashes, a torchrun is
    killed and leaves its ranks running.
    """
    procs = [p for p in sim.procs if any(r.state in ('starting', 'joined') for r in p.ranks)]
    if not procs:
        return
    proc = sim.rng.choice(procs)
    starting = [r for r in proc.ranks if r.state == 'starting']
    joined = [r for r in proc.ranks if r.state == 'joined']
    event = sim.rng.choices(['join', 'train', 'crash', 'kill_agent'], [5, 3, 1, 1])[0]
    if event == 'join' and starting and (not starting[0].late or sim.rng.random() < 0.15):
        if starting[0].try_join() == 'rejected':
            proc.rank_failed()
    elif event == 'train' and joined and not starting:
        proc.train_to_end(sim.rng.choices(['complete', 'partial', 'no_milestone'], [6, 3, 1])[0])
    elif event == 'crash' and joined:
        sim.rng.choice(joined).exit()
        proc.rank_failed()
    elif event == 'kill_agent' and proc.agent_alive():
        proc.returncode = -9


@pytest.mark.parametrize('seed', range(SEEDS))
def test_concurrent_workers_and_ranks(work_dir, quiet, monkeypatch, seed):
    rng = random.Random(seed)
    spec = _random_spec(work_dir, rng)
    root = _write_spec(spec)
    sim = Sim(root, rng)
    _patch_hooks(monkeypatch, sim)
    for _ in range(120):
        event = rng.choices(['worker', 'ranks', 'start', 'kill', 'stop', 'plan', 'probe'], [8, 8, 1, 1, 1, 1, 1])[0]
        if event == 'worker' and sim.workers:
            sim.step_worker(rng.choice(sim.workers))
        elif event == 'ranks':
            _rank_step(sim)
        elif event == 'start' and len(sim.workers) < 3:
            sim.start_worker()
        elif event == 'kill' and sim.workers:
            sim.kill_worker(rng.choice(sim.workers))
        elif event == 'stop' and sim.workers:
            sim.stop_worker(rng.choice(sim.workers))
        elif event == 'plan':
            spec = sim.try_replan(spec)
        elif event == 'probe':  # sweep.py status, it must not get in anyone's way
            sweep.JobState(str(root), rng.choice(sim.plan()['jobs'])).attempt_gone()
        sim.check()
    # the chaos stops: every job process runs to its end, workers stop, then one worker finishes the sweep
    for proc in sim.procs:
        while any(r.state == 'starting' for r in proc.ranks):
            rank = next(r for r in proc.ranks if r.state == 'starting')
            if rank.try_join() == 'rejected':
                proc.rank_failed()
            elif rank.state == 'waiting':
                raise Violation('a rank waits on a lock with every worker idle')
        if any(r.state == 'joined' for r in proc.ranks):
            proc.train_to_end('complete')
        sim.check()
    for worker in list(sim.workers):
        sim.kill_worker(worker)
    rng.random = lambda: 1.  # no more chaos
    sim.start_worker()
    for _ in range(500):
        sim.step_worker(sim.workers[0])
        for proc in sim.procs:
            for rank in proc.ranks:
                if rank.state == 'starting':
                    assert rank.try_join() == 'joined'
            if proc.agent_alive() and all(r.state == 'joined' for r in proc.ranks):
                proc.train_to_end('complete')
        executor = sim.workers[0][0]
        if not executor.running and not executor.ready_jobs():
            break
    sim.kill_worker(sim.workers[0])
    plan = sim.plan()
    states = {j['id']: sweep.JobState(str(root), j).status for j in plan['jobs']}
    assert set(states.values()) == {'completed'}, states
    _check_completed(root, plan)
    for path in (root / sweep.LOCK_DIR).glob('*'):
        if path.suffix in ('.lock', '.coord', '.train'):
            lock = sweep.Lock(str(path))
            assert lock.acquire(exclusive=True), f'{path} is still held'
            lock.release()
