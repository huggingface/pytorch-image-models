#!/usr/bin/env python3
""" timm Hyperparameter Sweep Orchestrator

Plan and run sweeps over train.py arguments, optionally with warmup-stable-decay (WSD) branching where one stable
trunk per hyperparameter combination feeds many independent cooldown branches resumed from its milestone checkpoints.

The planner writes one complete, effective train.py config per job (the same YAML --config accepts) and an immutable
plan.json with the job dependency graph, then the executor launches ready jobs on a pool of local GPUs. Nothing about
the search lives in train.py: every job is a plain `train.py --config` invocation that can also be run by hand.

Quick sweeps need no spec file, the search space comes from --sweep and the base config from the trailing train.py
arguments:

    python sweep.py run --name lr-wd --sweep lr=1e-4,3e-4,1e-3 --sweep weight_decay=0.05,0.1 --gpus 0,1,2,3 \\
        -- --model vit_small_patch16_224 --data-dir /data/imagenet --sched cosine --epochs 100 --amp

A YAML spec adds random search, paired axes, repeats and WSD branching:

    # sweep.yaml
    name: vit-wsd
    output: output/sweeps            # the sweep lives in output/sweeps/vit-wsd
    train:                           # base train.py config, same keys as a --config file
      config: configs/vit_small.yaml # optional base config file, keys below override it
      model: vit_small_patch16_224
      data_dir: /data/imagenet
      opt: adamw
      warmup_epochs: 5
    search:
      mode: grid                     # or random (then n_trials, and distributions below are allowed)
      seed: 42
      repeats: 1                     # > 1 repeats each trial with a different seed
      space:
        lr: [3e-4, 6e-4]             # a list is a grid axis (a 'choice' in random mode)
        weight_decay: {choice: [0.05, 0.1]}
        # mixup: {uniform: [0.0, 0.4]}, drop_path: {loguniform: [0.05, 0.3]}, epochs: {int: [80, 120]}
      zip:                           # paired axes, the lists are stepped together as one axis
        - {batch_size: [128, 256], lr_base_size: [128, 256]}
    metric:
      name: top1                     # a summary.csv eval_ metric, default is the train eval metric
      direction: maximize
    wsd:                             # omit for a plain sweep
      points: [40, 80, 120]          # completed stable epochs to branch cooldowns from
      decay_epochs: [8, 16]
      decay_fn: [1-sqrt, linear]
    execution:
      gpus_per_job: 1

    python sweep.py plan sweep.yaml                    # write configs + plan.json + graph.dot, run nothing
    python sweep.py run output/sweeps/vit-wsd --gpus 0,1,2,3
    python sweep.py status output/sweeps/vit-wsd
    python sweep.py results output/sweeps/vit-wsd

`run` on a sweep that was interrupted picks up where it stopped: finished jobs are skipped and a job with a partial
run resumes from its own last.pth.tar (train.py --resume auto). Workers on several machines sharing the sweep
directory can each `run` it, jobs are claimed atomically.

Paths in the generated configs and scripts are fixed when the plan is written, so a sweep directory can't be moved or
copied elsewhere and run from there: plan it again under the new location instead.

Hacked together by / Copyright 2026 Ross Wightman
"""
import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import random
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

import train as train_script
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import load_checkpoint_metadata

_logger = logging.getLogger('sweep')

PLAN_VERSION = 1
PLAN_FILE = 'plan.json'
STATE_DIR = 'state'
TIMM_ROOT = os.path.dirname(os.path.abspath(__file__))

# train.py arguments the planner or the executor owns, they can't be set in a base config or swept
RESERVED_KEYS = {
    'output', 'experiment', 'resume', 'stop_epoch', 'save_epochs',  # planner
    'local_rank', 'wandb_resume_id',  # host / executor
}
# additionally derived by the planner for every job of a WSD sweep
WSD_RESERVED_KEYS = {'sched', 'epochs', 'decay_epochs', 'lr_decay_fn'}
# left out of the logical config hash, they describe where a job runs rather than what it trains
UNHASHED_KEYS = {'output', 'experiment', 'resume', 'log_wandb', 'wandb_project', 'wandb_tags'}

SLUG_ABBREV = {
    'lr': 'lr', 'lr_base': 'lrb', 'weight_decay': 'wd', 'batch_size': 'bs', 'drop': 'drop', 'drop_path': 'dp',
    'smoothing': 'sm', 'mixup': 'mix', 'cutmix': 'cm', 'warmup_epochs': 'wu', 'epochs': 'ep', 'min_lr': 'minlr',
    'momentum': 'mom', 'clip_grad': 'cg', 'layer_decay': 'ld', 'seed': 'seed', 'model': '', 'opt': '', 'sched': '',
    'img_size': 'sz', 'reprob': 'rp', 'aa': 'aa', 'grad_accum_steps': 'ga', 'model_ema_decay': 'emad',
}


# --------------------------------------------------------------------------------------------------------------------
# spec -> trials
# --------------------------------------------------------------------------------------------------------------------

class SweepError(Exception):
    pass


def value_slug(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f'{value:g}'
    if isinstance(value, (list, tuple)):
        return 'x'.join(value_slug(v) for v in value)
    return ''.join(c if c.isalnum() or c in '.-' else '_' for c in str(value))


def trial_slug(params: Dict[str, Any], max_len: int = 48) -> str:
    parts = []
    for key, value in params.items():
        abbrev = SLUG_ABBREV.get(key, key.replace('_', ''))
        parts.append(f'{abbrev}{value_slug(value)}')
    slug = '-'.join(parts)
    return slug[:max_len].rstrip('-')


def config_hash(cfg: Dict[str, Any]) -> str:
    logical = {k: v for k, v in cfg.items() if k not in UNHASHED_KEYS}
    return hashlib.sha1(json.dumps(logical, sort_keys=True, default=str).encode()).hexdigest()[:8]


def parse_cli_value(text: str) -> Any:
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def numeric(value: Any) -> Any:
    """YAML 1.1 reads 1e-4 as a string, convert numeric strings the way the command line would."""
    if isinstance(value, str):
        for convert in (int, float):
            try:
                return convert(value)
            except ValueError:
                pass
    return value


def normalize_axis(name: str, spec: Any) -> Dict[str, Any]:
    """Normalize a search space entry to {'choice': [...]} or a {distribution: [lo, hi]} dict."""
    if isinstance(spec, dict):
        if len(spec) != 1:
            raise SweepError(f'search space "{name}" must be a list or a single {{distribution: args}} entry.')
        (dist, args), = spec.items()
        if dist == 'choice':
            if not isinstance(args, (list, tuple)) or not args:
                raise SweepError(f'search space "{name}" choice needs a non-empty list.')
            return {'choice': [numeric(v) for v in args]}
        if dist in ('uniform', 'loguniform', 'int'):
            if not (isinstance(args, (list, tuple)) and len(args) == 2):
                raise SweepError(f'search space "{name}" {dist} needs [low, high].')
            lo, hi = numeric(args[0]), numeric(args[1])
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lo, hi)):
                raise SweepError(f'search space "{name}" {dist} needs numeric [low, high], got {args}.')
            if dist == 'loguniform' and not (0 < lo <= hi):
                raise SweepError(f'search space "{name}" loguniform needs 0 < low <= high.')
            if lo > hi:
                raise SweepError(f'search space "{name}" {dist} needs low <= high.')
            return {dist: [lo, hi]}
        raise SweepError(f'search space "{name}" has unknown distribution "{dist}".')
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise SweepError(f'search space "{name}" is an empty list.')
        return {'choice': [numeric(v) for v in spec]}
    return {'choice': [numeric(spec)]}


def normalize_zip_groups(groups: Any) -> List[Dict[str, List[Any]]]:
    if not groups:
        return []
    if isinstance(groups, dict):
        groups = [groups]
    normalized = []
    for group in groups:
        if not isinstance(group, dict) or len(group) < 1:
            raise SweepError('each zip group must be a mapping of argument name -> list of values.')
        lengths = set()
        for key, values in group.items():
            if not isinstance(values, (list, tuple)) or not values:
                raise SweepError(f'zip group entry "{key}" must be a non-empty list.')
            lengths.add(len(values))
        if len(lengths) != 1:
            raise SweepError(f'zip group {sorted(group)} lists must have equal length, got {sorted(lengths)}.')
        normalized.append({k: [numeric(x) for x in v] for k, v in group.items()})
    return normalized


def sample_axis(rng: random.Random, axis: Dict[str, Any]) -> Any:
    (dist, args), = axis.items()
    if dist == 'choice':
        return rng.choice(args)
    lo, hi = args
    if dist == 'uniform':
        return rng.uniform(lo, hi)
    if dist == 'loguniform':
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    if dist == 'int':
        return rng.randint(int(lo), int(hi))
    raise SweepError(f'unknown distribution {dist}')


def expand_search(search: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a search block into an ordered list of parameter dicts (one per trial, before repeats)."""
    mode = search.get('mode', 'grid')
    if mode not in ('grid', 'random'):
        raise SweepError(f'search.mode must be grid or random, got {mode}.')
    space = {k: normalize_axis(k, v) for k, v in (search.get('space') or {}).items()}
    zip_groups = normalize_zip_groups(search.get('zip'))
    for group in zip_groups:
        clash = set(group) & set(space)
        if clash:
            raise SweepError(f'{sorted(clash)} appear in both search.space and search.zip.')
    if not space and not zip_groups:
        return [{}]

    if mode == 'grid':
        axes = []
        for key, axis in space.items():
            if 'choice' not in axis:
                raise SweepError(f'search space "{key}" uses a distribution, only lists are valid for grid search.')
            axes.append([{key: v} for v in axis['choice']])
        for group in zip_groups:
            n = len(next(iter(group.values())))
            axes.append([{k: v[i] for k, v in group.items()} for i in range(n)])
        trials = []
        for combo in itertools.product(*axes):
            params = {}
            for part in combo:
                params.update(part)
            trials.append(params)
        return trials

    n_trials = search.get('n_trials')
    if not n_trials or n_trials < 1:
        raise SweepError('search.n_trials must be a positive integer for random search.')
    rng = random.Random(search.get('seed', 42))
    trials = []
    for _ in range(n_trials):
        params = {k: sample_axis(rng, axis) for k, axis in space.items()}
        for group in zip_groups:
            i = rng.randrange(len(next(iter(group.values()))))
            params.update({k: v[i] for k, v in group.items()})
        trials.append(params)
    return trials


# --------------------------------------------------------------------------------------------------------------------
# spec -> plan
# --------------------------------------------------------------------------------------------------------------------

def check_reserved(keys, where: str, extra_reserved=()):
    reserved = sorted((set(keys) & (RESERVED_KEYS | set(extra_reserved))))
    if reserved:
        raise SweepError(f'{reserved} in {where} are set by the sweep planner and cannot be specified directly.')


def validate_train_keys(cfg: Dict[str, Any], where: str):
    """Validate against train.py's parser, unknown keys are an error here (a sweep must not silently ignore them)."""
    actions = train_script.config_arguments()
    unknown = sorted(k for k in cfg if k not in actions)
    if unknown:
        raise SweepError(f'{unknown} in {where} match no train.py argument.')
    try:
        train_script.validate_config(cfg)
    except SystemExit as e:  # argparse error
        raise SweepError(f'invalid value in {where}, see the error above.') from e


def build_base_config(train_block: Dict[str, Any], spec_dir: str, wsd: bool = False) -> Dict[str, Any]:
    """Full effective train.py config: parser defaults <- sweep defaults <- optional base config file <- train block."""
    train_block = dict(train_block or {})
    cfg = vars(train_script.parser.parse_args([]))
    if wsd:
        cfg['sched_on_updates'] = True  # short decays are poorly sampled per epoch, any explicit setting overrides
    base_path = train_block.pop('config', None)
    if base_path:
        base_path = base_path if os.path.isabs(base_path) else os.path.join(spec_dir, base_path)
        base_cfg = train_script.load_config(base_path)
        check_reserved(base_cfg, f'base config {base_path}')
        cfg.update(base_cfg)
    check_reserved(train_block, 'the train block')
    validate_train_keys(train_block, 'the train block')
    cfg.update(train_block)
    return cfg


def resolve_metric(spec: Dict[str, Any], base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    metric = dict(spec.get('metric') or {})
    task_default = 'map' if base_cfg.get('task') == 'multilabel' else 'top1'
    name = metric.get('name') or base_cfg.get('eval_metric') or task_default
    direction = metric.get('direction') or ('minimize' if 'loss' in name else 'maximize')
    if direction not in ('maximize', 'minimize'):
        raise SweepError(f'metric.direction must be maximize or minimize, got {direction}.')
    return {'name': name, 'direction': direction}


def schedule_horizon(point: int, decay_epochs: int, cfg: Dict[str, Any]) -> int:
    """`--epochs` that places a `decay_epochs` long WSD decay right after `point` completed epochs."""
    return point + decay_epochs - (cfg['warmup_epochs'] if cfg['warmup_prefix'] else 0)


def total_epochs(cfg: Dict[str, Any]) -> int:
    """Epochs train.py will run for a config, computed by the scheduler factory train.py uses.

    Cycle schedulers multiply the horizon by their cycle count, some add cooldown epochs and some don't, so the
    factory is asked rather than re-deriving its rules. A throwaway optimizer stands in for the real one.
    """
    ns = argparse.Namespace(**cfg)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    eval_metric = cfg.get('eval_metric') or 'top1'
    # per-update schedules get a large synthetic update count so fractional decays don't round to zero, the factory
    # divides the total back to whole epochs
    updates_per_epoch = 1_000_000 if cfg.get('sched_on_updates') else 1
    try:
        _, num_epochs = create_scheduler_v2(
            optimizer, **scheduler_kwargs(ns, decreasing_metric=eval_metric == 'loss'),
            updates_per_epoch=updates_per_epoch)
    except (ValueError, AssertionError) as e:
        raise SweepError(f'invalid schedule configuration: {e}') from e
    return num_epochs


def normalize_wsd(wsd: Dict[str, Any], base_cfg: Dict[str, Any], train_block: Dict[str, Any]) -> Dict[str, Any]:
    points = wsd.get('points')
    if not points or not all(isinstance(p, int) and p > 0 for p in points):
        raise SweepError('wsd.points must be a non-empty list of positive completed epoch counts.')
    points = sorted(set(points))
    decay_epochs = wsd.get('decay_epochs') or [max(1, round(0.2 * points[-1]))]
    if not all(isinstance(d, int) and d > 0 for d in decay_epochs):
        raise SweepError('wsd.decay_epochs must be positive integers.')
    decay_fn = wsd.get('decay_fn') or ['1-sqrt']
    if isinstance(decay_fn, str):
        decay_fn = [decay_fn]
    min_lr = wsd.get('min_lr')
    if min_lr is not None and not isinstance(min_lr, (list, tuple)):
        min_lr = [min_lr]
    sched = train_block.get('sched', base_cfg.get('sched'))
    if 'sched' in train_block and sched != 'wsd' or sched not in (None, 'cosine', 'wsd'):  # cosine: parser default
        raise SweepError(f'a WSD sweep sets sched=wsd, the train config has sched={sched}.')
    if base_cfg.get('train_size_schedule') == 'progressive':
        raise SweepError('WSD branches change --epochs, which also drives --train-size-schedule progressive. '
                         'Use a constant size schedule.')
    if base_cfg.get('no_resume_opt'):
        raise SweepError('WSD cooldowns must restore optimizer state, remove no_resume_opt.')
    return {'points': points, 'decay_epochs': sorted(set(decay_epochs)), 'decay_fn': list(decay_fn), 'min_lr': min_lr}


def make_job(
        job_id: str,
        trial_id: str,
        kind: str,
        job_dir: str,
        cfg: Dict[str, Any],
        config_name: str = 'config.yaml',
        deps: Optional[List[str]] = None,
        inputs: Optional[List[str]] = None,
        gpus: int = 1,
        end_epoch: Optional[int] = None,
        wsd: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        'id': job_id,
        'trial': trial_id,
        'kind': kind,
        'dir': job_dir,
        'config': os.path.join(job_dir, config_name),
        'script': os.path.join(job_dir, config_name.replace('config', 'run').replace('.yaml', '.sh')),
        'config_hash': config_hash(cfg),
        'deps': deps or [],
        'inputs': inputs or [],
        'gpus': gpus,
        'end_epoch': end_epoch if end_epoch is not None else total_epochs(cfg),
        'wsd': wsd,
        # checkpoints that must exist, at or past end_epoch, for the job to count as complete
        'artifacts': [os.path.join(job_dir, 'last.pth.tar')] + list(artifacts or []),
        '_cfg': cfg,  # written to config.yaml, stripped from plan.json
    }


def plan_sweep(spec: Dict[str, Any], spec_dir: str = '.') -> Dict[str, Any]:
    """Expand a sweep spec into trials and jobs. Pure, writes nothing."""
    name = spec.get('name')
    if not name:
        raise SweepError('the sweep needs a name.')
    output = spec.get('output') or os.path.join('output', 'sweeps')
    root = os.path.abspath(os.path.join(spec_dir, output, name) if not os.path.isabs(output) else
                           os.path.join(output, name))
    wsd_spec = spec.get('wsd')
    base_cfg = build_base_config(spec.get('train'), spec_dir, wsd=bool(wsd_spec))
    search = dict(spec.get('search') or {})
    space_keys = set(search.get('space') or {}) | {k for g in normalize_zip_groups(search.get('zip')) for k in g}
    check_reserved(space_keys, 'the search space', WSD_RESERVED_KEYS if wsd_spec else ())
    check_reserved(spec.get('train') or {}, 'the train block', WSD_RESERVED_KEYS - {'sched'} if wsd_spec else ())
    validate_train_keys({k: None for k in space_keys}, 'the search space')
    repeats = int(search.get('repeats', 1) or 1)
    if repeats > 1 and 'seed' in space_keys:
        raise SweepError('use either search.repeats or a seed axis, not both.')
    execution = dict(spec.get('execution') or {})
    gpus_per_job = int(execution.get('gpus_per_job', 1) or 1)
    metric = resolve_metric(spec, base_cfg)
    wsd = normalize_wsd(wsd_spec, base_cfg, spec.get('train') or {}) if wsd_spec else None

    trials, jobs = [], []
    for index, params in enumerate(expand_search(search)):
        validate_train_keys(params, f'trial {index} parameters')
        for repeat in range(repeats):
            trial_params = dict(params)
            slug = trial_slug(trial_params)
            if repeats > 1:
                trial_params['seed'] = base_cfg['seed'] + repeat
            trial_id = f'trial-{index:03d}' + (f'-{slug}' if slug else '') + (f'-r{repeat}' if repeats > 1 else '')
            trial_cfg = dict(base_cfg)
            trial_cfg.update(trial_params)
            trial_dir = os.path.join(root, trial_id)
            if wsd is not None and wsd['points'][0] <= trial_cfg['warmup_epochs']:
                raise SweepError(f'the first WSD point ({wsd["points"][0]}) must be after the warmup '
                                 f'({trial_cfg["warmup_epochs"]} epochs in {trial_id}).')
            trial = {
                'id': trial_id, 'index': index, 'repeat': repeat, 'params': trial_params, 'jobs': [],
            }
            trials.append(trial)

            if wsd is None:
                cfg = dict(trial_cfg, output=trial_dir, experiment='train', resume='auto')
                job = make_job(f'{trial_id}/train', trial_id, 'train', os.path.join(trial_id, 'train'), cfg,
                               gpus=gpus_per_job)
                jobs.append(job)
                trial['jobs'].append(job['id'])
                continue

            points = wsd['points']
            trunk_cfg = dict(
                trial_cfg, sched='wsd', epochs=schedule_horizon(points[-1], 1, trial_cfg), decay_epochs=1,
                lr_decay_fn=trial_cfg.get('lr_decay_fn') or wsd['decay_fn'][0],
                save_epochs=list(points), output=trial_dir, experiment='stable', resume='auto',
            )
            prev_job = None
            stable_dir = os.path.join(trial_id, 'stable')
            for point in points:
                cfg = dict(trunk_cfg, stop_epoch=point)
                job = make_job(
                    f'{trial_id}/stable-{point}', trial_id, 'wsd_stable', stable_dir, cfg,
                    config_name=f'config-{point}.yaml', deps=[prev_job] if prev_job else [], gpus=gpus_per_job,
                    end_epoch=point, wsd={'point': point},
                    artifacts=[os.path.join(stable_dir, f'milestone-{point}.pth.tar')],
                )
                jobs.append(job)
                trial['jobs'].append(job['id'])
                prev_job = job['id']
                for decay_epochs in wsd['decay_epochs']:
                    for decay_fn in wsd['decay_fn']:
                        for min_lr in (wsd['min_lr'] or [None]):
                            branch = f'decay-p{point}-d{decay_epochs}-{value_slug(decay_fn)}'
                            branch_wsd = {'point': point, 'decay_epochs': decay_epochs, 'decay_fn': decay_fn}
                            cfg = dict(
                                trial_cfg, sched='wsd', epochs=schedule_horizon(point, decay_epochs, trial_cfg),
                                decay_epochs=decay_epochs, lr_decay_fn=decay_fn, output=trial_dir,
                                experiment=branch, stop_epoch=None, save_epochs=[],
                                resume=os.path.join(trial_dir, 'stable', f'milestone-{point}.pth.tar'),
                            )
                            if min_lr is not None:
                                branch += f'-minlr{value_slug(min_lr)}'
                                branch_wsd['min_lr'] = min_lr
                                cfg['min_lr'] = min_lr
                                cfg['experiment'] = branch
                            bjob = make_job(
                                f'{trial_id}/{branch}', trial_id, 'wsd_cooldown', os.path.join(trial_id, branch), cfg,
                                deps=[job['id']], inputs=[os.path.join(stable_dir, f'milestone-{point}.pth.tar')],
                                gpus=gpus_per_job, end_epoch=point + decay_epochs + trial_cfg['cooldown_epochs'],
                                wsd=branch_wsd,
                            )
                            jobs.append(bjob)
                            trial['jobs'].append(bjob['id'])

    ids = [j['id'] for j in jobs]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise SweepError(f'duplicate job ids {dupes[:3]}, make the swept values distinguishable.')
    return {
        'version': PLAN_VERSION,
        'name': name,
        'root': root,
        'created': datetime.now().isoformat(timespec='seconds'),
        'python': sys.executable,
        'train_script': os.path.join(TIMM_ROOT, 'train.py'),
        'metric': metric,
        'search': {k: search.get(k) for k in ('mode', 'seed', 'n_trials', 'repeats') if k in search},
        'space_keys': sorted(space_keys),
        'wsd': wsd,
        'gpus_per_job': gpus_per_job,
        'trials': trials,
        'jobs': jobs,
    }


# --------------------------------------------------------------------------------------------------------------------
# plan -> files
# --------------------------------------------------------------------------------------------------------------------

def run_script_text(job: Dict[str, Any], plan: Dict[str, Any]) -> str:
    config_path = os.path.join(plan['root'], job['config'])
    launcher = '"${PYTHON:-python}"'
    if job['gpus'] > 1:
        launcher += f' -m torch.distributed.run --standalone --nproc_per_node={job["gpus"]}'
    return (
        '#!/bin/bash\n'
        f'# {job["id"]} ({job["kind"]}), generated by sweep.py for sweep "{plan["name"]}"\n'
        '# Extra arguments are passed through to train.py, e.g. --resume auto to continue an interrupted run.\n'
        'set -e\n'
        f'cd "{TIMM_ROOT}"\n'
        f'exec {launcher} train.py --config "{config_path}" "$@"\n'
    )


def graph_dot(plan: Dict[str, Any]) -> str:
    lines = ['digraph sweep {', '  rankdir=LR;', '  node [shape=box, fontsize=10];']
    for trial in plan['trials']:
        lines.append(f'  subgraph "cluster_{trial["id"]}" {{')
        lines.append(f'    label="{trial["id"]}";')
        for job_id in trial['jobs']:
            lines.append(f'    "{job_id}" [label="{job_id.split("/", 1)[1]}"];')
        lines.append('  }')
    for job in plan['jobs']:
        for dep in job['deps']:
            label = f' [label="{os.path.basename(job["inputs"][0])}"]' if job['inputs'] else ''
            lines.append(f'  "{dep}" -> "{job["id"]}"{label};')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def plan_signature(plan: Dict[str, Any]) -> Dict[str, str]:
    return {j['id']: j['config_hash'] for j in plan['jobs']}


def write_plan(plan: Dict[str, Any], spec: Dict[str, Any], force: bool = False) -> str:
    root = plan['root']
    plan_path = os.path.join(root, PLAN_FILE)
    if os.path.exists(plan_path):
        existing = load_plan(root)
        old_sig, new_sig = plan_signature(existing), plan_signature(plan)
        if old_sig != new_sig:
            if not force:
                raise SweepError(
                    f'{plan_path} exists with a different set of jobs or configs. Use a new sweep name, or --force to '
                    f'overwrite the plan (jobs whose config changed are reset, their old results are set aside).')
            _logger.warning(f'Overwriting the existing plan in {root}.')
        else:
            _logger.info(f'Plan in {root} is unchanged.')
    invalidate_changed_jobs(plan)
    os.makedirs(os.path.join(root, STATE_DIR), exist_ok=True)
    for job in plan['jobs']:
        job_dir = os.path.join(root, job['dir'])
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(root, job['config']), 'w') as f:
            f.write(f'# {job["id"]} ({job["kind"]}), effective train.py config generated by sweep.py\n')
            f.write(f'# config_hash: {job["config_hash"]}\n')
            yaml.safe_dump(job['_cfg'], f, default_flow_style=False, sort_keys=False)
        script_path = os.path.join(root, job['script'])
        with open(script_path, 'w') as f:
            f.write(run_script_text(job, plan))
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    with open(os.path.join(root, 'sweep.yaml'), 'w') as f:
        yaml.safe_dump(spec, f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(root, 'graph.dot'), 'w') as f:
        f.write(graph_dot(plan))
    serializable = dict(plan, jobs=[{k: v for k, v in j.items() if k != '_cfg'} for j in plan['jobs']])
    tmp_path = plan_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    os.replace(tmp_path, plan_path)
    return root


def recorded_config_hash(config_path: str) -> Optional[str]:
    """The config hash sweep.py wrote into a generated config file's header, None if absent."""
    try:
        with open(config_path) as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if line.startswith('# config_hash:'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return None


def invalidate_changed_jobs(plan: Dict[str, Any]):
    """Reset jobs whose directory holds outputs of a different config, so stale results are never reported under
    the new config and --resume auto can't continue the old run.

    Identity comes from the hash recorded in each directory's existing config file, not from the previous plan, so a
    job removed and later reintroduced with another config is caught too.
    """
    root = plan['root']
    # read every recorded hash first: WSD trunk segments share a directory, moving it for one segment would hide the
    # config files of the others
    old_hashes = {job['id']: recorded_config_hash(os.path.join(root, job['config'])) for job in plan['jobs']}
    changed = [j for j in plan['jobs'] if old_hashes[j['id']] not in (None, j['config_hash'])]
    moved = set()
    for job in changed:
        job_dir = os.path.join(root, job['dir'])
        has_outputs = any(os.path.exists(os.path.join(job_dir, f)) for f in ('summary.csv', 'last.pth.tar', 'run.log'))
        if job_dir in moved or not has_outputs:
            continue
        old_hash = old_hashes[job['id']]
        stale_dir = f'{job_dir}.stale-{old_hash}'
        n = 1
        while os.path.exists(stale_dir):
            n += 1
            stale_dir = f'{job_dir}.stale-{old_hash}-{n}'
        os.rename(job_dir, stale_dir)
        moved.add(job_dir)
        _logger.warning(f'{job["id"]} config changed, previous outputs moved to {stale_dir}')
    # reset every changed job, and every job whose outputs went with a moved directory
    for job in plan['jobs']:
        if job in changed or os.path.join(root, job['dir']) in moved:
            state = JobState(root, job)
            state.release()
            if os.path.exists(state.path):
                os.remove(state.path)


def load_plan(root: str) -> Dict[str, Any]:
    plan_path = os.path.join(root, PLAN_FILE)
    if not os.path.isfile(plan_path):
        raise SweepError(f'{root} has no {PLAN_FILE}, run `sweep.py plan` first.')
    with open(plan_path) as f:
        plan = json.load(f)
    if plan.get('version') != PLAN_VERSION:
        raise SweepError(f'{plan_path} is plan version {plan.get("version")}, this sweep.py reads {PLAN_VERSION}.')
    plan['root'] = os.path.abspath(root)
    return plan


# --------------------------------------------------------------------------------------------------------------------
# job state
# --------------------------------------------------------------------------------------------------------------------

def summary_epochs(root: str, job: Dict[str, Any]) -> Optional[int]:
    """Completed epochs recorded in the job's summary.csv, None if there is no summary yet."""
    path = os.path.join(root, job['dir'], 'summary.csv')
    if not os.path.isfile(path):
        return None
    last = None
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                last = int(float(row['epoch']))
            except (KeyError, ValueError, TypeError):
                continue
    return None if last is None else last + 1


def checkpoint_epochs(path: str) -> Optional[int]:
    """Completed epochs stored in a train.py checkpoint, None if it can't be read."""
    try:
        checkpoint = load_checkpoint_metadata(path)
    except Exception as e:  # a partially written or foreign file
        _logger.warning(f'Could not read checkpoint {path}: {e}')
        return None
    if not isinstance(checkpoint, dict) or 'epoch' not in checkpoint:
        return None
    return checkpoint['epoch'] + (1 if checkpoint.get('version', 1) > 1 else 0)


class JobState:
    """Per-job status record under <root>/state, plus an atomic claim directory while a worker runs the job."""

    def __init__(self, root: str, job: Dict[str, Any]):
        self.root = root
        self.job = job
        flat = job['id'].replace('/', '__')
        self.path = os.path.join(root, STATE_DIR, flat + '.json')
        self.claim_dir = os.path.join(root, STATE_DIR, flat + '.claim')
        self.reload()

    def reload(self):
        """Re-read the state file, other workers sharing the sweep directory may have changed it."""
        self.data = {'id': self.job['id'], 'status': 'pending', 'attempts': []}
        if os.path.isfile(self.path):
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except (OSError, json.JSONDecodeError):
                _logger.warning(f'Unreadable state file {self.path}, treating {self.job["id"]} as pending.')

    @property
    def status(self) -> str:
        return self.data['status']

    @property
    def attempts(self) -> List[Dict[str, Any]]:
        return self.data['attempts']

    def save(self):
        tmp = self.path + f'.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def claim(self) -> bool:
        try:
            os.mkdir(self.claim_dir)
        except FileExistsError:
            return False
        with open(os.path.join(self.claim_dir, 'owner'), 'w') as f:
            f.write(f'{socket.gethostname()}:{os.getpid()}\n')
        return True

    def release(self):
        shutil.rmtree(self.claim_dir, ignore_errors=True)

    def claim_owner(self) -> Optional[Tuple[str, int]]:
        try:
            with open(os.path.join(self.claim_dir, 'owner')) as f:
                host, pid = f.read().strip().rsplit(':', 1)
            return host, int(pid)
        except (OSError, ValueError):
            return None

    def set(self, status: str, **attempt_update):
        self.data['status'] = status
        if attempt_update and self.attempts:
            self.attempts[-1].update(attempt_update)
        self.save()

    def start_attempt(self, pid: int, gpus: List[int], command: List[str]):
        self.attempts.append({
            'attempt': len(self.attempts) + 1,
            'host': socket.gethostname(),
            'pid': pid,
            'gpus': gpus,
            'command': command,
            'start': datetime.now().isoformat(timespec='seconds'),
        })
        self.data['status'] = 'running'
        self.save()


def short_job_id(job_id: str) -> str:
    """'trial-000-lr0.001-wd0.05/stable-18' -> 'trial-000/stable-18' for console prefixes."""
    trial, _, job = job_id.partition('/')
    parts = trial.split('-')
    if len(parts) >= 2 and parts[0] == 'trial':
        trial = '-'.join(parts[:2]) + ('-' + parts[-1] if parts[-1].startswith('r') and parts[-1][1:].isdigit() else '')
    return f'{trial}/{job}' if job else trial


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------------------------------------------------
# executor
# --------------------------------------------------------------------------------------------------------------------

class Executor:
    def __init__(
            self,
            plan: Dict[str, Any],
            gpus: Optional[List[int]] = None,
            max_parallel: Optional[int] = None,
            retries: int = 0,
            poll_interval: float = 10.,
            dry_run: bool = False,
            extra_args: Optional[List[str]] = None,
            echo: bool = True,
    ):
        self.plan = plan
        self.root = plan['root']
        self.jobs = OrderedDict((j['id'], j) for j in plan['jobs'])
        self.states = {j['id']: JobState(self.root, j) for j in plan['jobs']}
        self.gpus = gpus
        if max_parallel is None:
            max_parallel = len(gpus) if gpus else 1
        self.max_parallel = max_parallel
        self.retries = retries
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.extra_args = extra_args or []
        self.echo = echo  # mirror job output to the console (always written to the job's run.log)
        self.running: Dict[str, Tuple[subprocess.Popen, List[int], Any, threading.Thread]] = {}
        self.free_gpus = list(gpus) if gpus else []
        self.launched_attempts: Dict[str, int] = {}

    # -- reconciliation with what is on disk

    def reconcile(self):
        """Bring state files in line with the filesystem: finished jobs are completed, dead runners are failed."""
        for job_id, job in self.jobs.items():
            state = self.states[job_id]
            if state.status == 'completed' or job_id in self.running:
                continue  # a job this worker is running is settled by finish(), its files may still be in flight
            if state.status == 'running':
                owner = state.claim_owner()
                last = state.attempts[-1] if state.attempts else {}
                host = owner[0] if owner else last.get('host')
                # the claim records the orchestrator, the attempt records the training process: either alive
                # means the job is still in progress (an orchestrator can die and leave its child training)
                pids = [p for p in ((owner[1] if owner else None), last.get('pid')) if p]
                if host == socket.gethostname() and any(pid_alive(p) for p in pids):
                    continue
                if host != socket.gethostname() and owner is not None:
                    continue  # a worker elsewhere holds the claim, don't steal it
                _logger.warning(f'{job_id} was running (pids {pids} on {host}) but is gone, marking interrupted.')
                state.release()
                state.set('interrupted', end=datetime.now().isoformat(timespec='seconds'))
            self.recover_artifacts(job)
            if self.on_disk_complete(job):
                _logger.info(f'{job_id} already has {job["end_epoch"]} epochs on disk, marking completed.')
                state.release()
                state.set('completed', end=datetime.now().isoformat(timespec='seconds'), reconciled=True)

    def refresh_states(self):
        """Pick up state changes made by other workers for jobs this worker isn't running."""
        for job_id, state in self.states.items():
            if job_id not in self.running:
                state.reload()

    def recover_artifacts(self, job: Dict[str, Any]):
        """Recreate a missing milestone from a checkpoint already at its epoch, as train.py would have.

        If the milestone save failed after the final epoch, a retry would resume at the stop epoch, train nothing and
        never produce it. `last.pth.tar` (if it hasn't moved on) or the ranked `checkpoint-N` for that epoch hold the
        identical state.
        """
        job_dir = os.path.join(self.root, job['dir'])
        for rel in job.get('artifacts', []):
            path = os.path.join(self.root, rel)
            name = os.path.basename(path)
            if os.path.isfile(path) or not name.startswith('milestone-'):
                continue
            point = int(name[len('milestone-'):].split('.')[0])
            for candidate in ('last.pth.tar', f'checkpoint-{point - 1}.pth.tar'):
                source = os.path.join(job_dir, candidate)
                if os.path.isfile(source) and checkpoint_epochs(source) == point:
                    try:
                        os.link(source, path)
                    except OSError:
                        shutil.copy2(source, path)
                    _logger.warning(f'{job["id"]}: recreated missing {name} from {candidate}')
                    break

    def on_disk_complete(self, job: Dict[str, Any]) -> bool:
        """The summary has reached the end epoch and so have the job's checkpoints.

        train.py writes the summary row before the checkpoint, so the summary alone can be one epoch ahead of a
        run that died (or ran out of disk) during its final save.
        """
        epochs = summary_epochs(self.root, job)
        if epochs is None or epochs < job['end_epoch']:
            return False
        for rel in job.get('artifacts', []):
            path = os.path.join(self.root, rel)
            if not os.path.isfile(path):
                return False
            saved = checkpoint_epochs(path)
            if saved is None or saved < job['end_epoch']:
                return False
        return True

    # -- readiness

    def deps_status(self, job: Dict[str, Any]) -> str:
        statuses = [self.states[d].status for d in job['deps']]
        if any(s in ('failed', 'blocked') for s in statuses):
            return 'blocked'
        if all(s == 'completed' for s in statuses):
            return 'ready'
        return 'waiting'

    def is_ready(self, job: Dict[str, Any]) -> bool:
        state = self.states[job['id']]
        if state.status in ('completed', 'running'):
            return False
        if state.status == 'failed' and self.launched_attempts.get(job['id'], 0) > self.retries:
            return False
        if self.deps_status(job) != 'ready':
            return False
        return self.dry_run or all(os.path.exists(os.path.join(self.root, p)) for p in job['inputs'])

    def ready_jobs(self) -> List[Dict[str, Any]]:
        return [j for j in self.jobs.values() if self.is_ready(j)]

    # -- launching

    def launch(self, job: Dict[str, Any], gpus: List[int]) -> bool:
        """Start a job, returning False if nothing was started (dry run, lost claim, already done elsewhere)."""
        state = self.states[job['id']]
        job_dir = os.path.join(self.root, job['dir'])
        command = [os.path.join(self.root, job['script'])] + list(self.extra_args)
        if os.path.exists(os.path.join(job_dir, 'last.pth.tar')) and '--resume' not in command:
            command += ['--resume', 'auto']  # pick the job's own partial run back up
        env = dict(os.environ)
        if self.gpus is not None:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpus)
        env.setdefault('PYTHON', self.plan.get('python') or sys.executable)
        env.setdefault('PYTHONUNBUFFERED', '1')  # prompt pass-through of the job's console output
        if self.dry_run:
            _logger.info(f'[dry run] {job["id"]} ({job["gpus"]} gpu): {" ".join(command)}')
            return False
        if not state.claim():
            owner = state.claim_owner()
            _logger.info(f'{job["id"]} is claimed by {owner}, skipping.')
            return False
        # under the claim, make sure another worker didn't finish or start it since our last look
        state.reload()
        if state.status == 'running':
            state.release()
            _logger.info(f'{job["id"]} is being run by another worker, skipping.')
            return False
        self.recover_artifacts(job)
        if state.status == 'completed' or self.on_disk_complete(job):
            if state.status != 'completed':
                state.set('completed', end=datetime.now().isoformat(timespec='seconds'), reconciled=True)
            state.release()
            _logger.info(f'{job["id"]} was finished by another worker, skipping.')
            return False
        attempt = len(state.attempts) + 1
        log_path = os.path.join(job_dir, 'run.log')
        log = open(log_path, 'a')
        log.write(f'\n===== sweep.py attempt {attempt} {datetime.now().isoformat(timespec="seconds")} '
                  f'{" ".join(command)}\n')
        log.flush()
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=TIMM_ROOT,
            text=True, errors='replace', bufsize=1,
        )
        reader = threading.Thread(target=self._tee, args=(proc, log, short_job_id(job['id'])), daemon=True)
        reader.start()
        state.start_attempt(proc.pid, gpus, command)
        self.launched_attempts[job['id']] = self.launched_attempts.get(job['id'], 0) + 1
        self.running[job['id']] = (proc, gpus, log, reader)
        _logger.info(f'Started {job["id"]} (attempt {attempt}, pid {proc.pid}'
                     f'{", gpus " + str(gpus) if gpus else ""}), log: {log_path}')
        return True

    def _tee(self, proc: subprocess.Popen, log, prefix: str):
        """Copy a job's output to its run.log and, unless quiet, to the console with a job prefix."""
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if self.echo:
                sys.stdout.write(f'[{prefix}] {line}')
                sys.stdout.flush()

    def finish(self, job_id: str, returncode: int):
        proc, gpus, log, reader = self.running.pop(job_id)
        reader.join(timeout=10)
        log.close()
        self.free_gpus.extend(gpus)
        job, state = self.jobs[job_id], self.states[job_id]
        end = datetime.now().isoformat(timespec='seconds')
        # publish the terminal state before releasing the claim, so no other worker sees a claimable 'running' job
        if returncode != 0:
            log_path = os.path.join(self.root, job['dir'], 'run.log')
            _logger.error(f'{job_id} failed with exit code {returncode}, see {log_path}')
            state.set('failed', end=end, returncode=returncode)
        elif not self.on_disk_complete(job):
            epochs = summary_epochs(self.root, job)
            _logger.error(f'{job_id} exited cleanly but its summary has {epochs} of {job["end_epoch"]} epochs.')
            state.set('failed', end=end, returncode=returncode, reason='incomplete')
        else:
            _logger.info(f'Completed {job_id}')
            state.set('completed', end=end, returncode=returncode)
        state.release()

    def poll(self):
        for job_id in list(self.running):
            proc = self.running[job_id][0]
            code = proc.poll()
            if code is not None:
                self.finish(job_id, code)

    def terminate(self):
        for job_id, (proc, gpus, log, reader) in list(self.running.items()):
            _logger.warning(f'Stopping {job_id} (pid {proc.pid})')
            proc.terminate()
        deadline = time.time() + 30
        for job_id, (proc, gpus, log, reader) in list(self.running.items()):
            try:
                proc.wait(timeout=max(0., deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            reader.join(timeout=5)
            log.close()
            self.states[job_id].set('interrupted', end=datetime.now().isoformat(timespec='seconds'))
            self.states[job_id].release()
        self.running.clear()

    def run(self) -> Dict[str, int]:
        self.reconcile()
        try:
            while True:
                self.poll()
                if not self.dry_run:
                    self.refresh_states()
                    self.reconcile()  # notices jobs finished, or abandoned, by other workers
                for job in self.ready_jobs():
                    if len(self.running) >= self.max_parallel:
                        break
                    gpus: List[int] = []
                    if self.gpus is not None and not self.dry_run:
                        if len(self.free_gpus) < job['gpus']:
                            continue
                        gpus = [self.free_gpus.pop(0) for _ in range(job['gpus'])]
                    if not self.launch(job, gpus):
                        self.free_gpus.extend(gpus)
                    if self.dry_run:
                        self.states[job['id']].data['status'] = 'completed'  # pretend, to walk the whole graph
                if self.dry_run:
                    if not self.ready_jobs():
                        break
                    continue
                if not self.running and not self.ready_jobs():
                    elsewhere = [j for j, st in self.states.items() if st.status == 'running']
                    if not elsewhere:
                        break
                    _logger.debug(f'Waiting on jobs running elsewhere: {elsewhere}')
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            _logger.warning('Interrupted, stopping running jobs. Re-run to resume.')
            self.terminate()
        return self.counts()

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for job in self.jobs.values():
            status = self.states[job['id']].status
            if status in ('pending', 'interrupted', 'failed') and self.deps_status(job) == 'blocked':
                status = 'blocked'
            counts[status] = counts.get(status, 0) + 1
        return counts


# --------------------------------------------------------------------------------------------------------------------
# status / results
# --------------------------------------------------------------------------------------------------------------------

def print_table(rows: List[Dict[str, Any]], columns: List[str]):
    if not rows:
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ''))) for r in rows)) for c in columns}
    print('  '.join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print('  '.join(str(row.get(c, '')).ljust(widths[c]) for c in columns))


def show_status(plan: Dict[str, Any]):
    executor = Executor(plan)
    executor.reconcile()
    rows = []
    for job in plan['jobs']:
        state = executor.states[job['id']]
        status = state.status
        if status in ('pending', 'interrupted', 'failed') and executor.deps_status(job) == 'blocked':
            status = 'blocked'
        elif status == 'pending' and executor.is_ready(job):
            status = 'ready'
        epochs = min(summary_epochs(plan['root'], job) or 0, job['end_epoch'])
        rows.append({
            'job': job['id'], 'status': status, 'attempts': len(state.attempts),
            'epochs': f'{epochs}/{job["end_epoch"]}',
        })
    print_table(rows, ['job', 'status', 'attempts', 'epochs'])
    counts = executor.counts()
    print('\n' + ', '.join(f'{k}: {v}' for k, v in sorted(counts.items())))


def summary_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('epoch') not in (None, '', 'epoch')]
    by_epoch: Dict[int, Dict[str, Any]] = OrderedDict()
    for r in rows:
        by_epoch[int(float(r['epoch']))] = r  # a resumed run may repeat an epoch, keep the latest
    return list(by_epoch.values())


def metric_maximize(plan: Dict[str, Any], metric_col: str) -> bool:
    """Direction for the column results were collected on, which may be a loss fallback for the planned metric."""
    if metric_col.endswith(plan['metric']['name']):
        return plan['metric']['direction'] == 'maximize'
    return not metric_col.endswith('loss')


def collect_results(plan: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """One row per job with a summary, with the final and best values of the sweep metric.

    Without a validation set the summaries have no eval metrics; results then rank on train_loss so the sweep still
    reports, and the endpoint checkpoints can be evaluated separately (validate.py).
    """
    metric_name = plan['metric']['name']
    executor = Executor(plan)
    executor.reconcile()
    trials = {t['id']: t for t in plan['trials']}
    results, metric_col = [], None
    for job in plan['jobs']:
        path = os.path.join(plan['root'], job['dir'], 'summary.csv')
        if not os.path.isfile(path):
            continue
        rows = summary_rows(path)
        if not rows:
            continue
        if metric_col is None:
            for candidate in (f'eval_{metric_name}', f'train_{metric_name}', metric_name):
                if candidate in rows[-1]:
                    metric_col = candidate
                    break
            if metric_col is None and 'train_loss' in rows[-1]:
                _logger.warning(
                    f'No {metric_name} in {path} (no validation set?), ranking by train_loss instead. Evaluate the '
                    f'endpoint checkpoints with validate.py to compare on the intended metric.')
                metric_col = 'train_loss'
            if metric_col is None:
                raise SweepError(f'metric {metric_name} not found in {path}, columns: {list(rows[-1])}')
            maximize = metric_maximize(plan, metric_col)
        # WSD trunk segments share one summary, a segment's rows are those up to its own end epoch
        values = [
            (float(r[metric_col]), int(float(r['epoch']))) for r in rows
            if r.get(metric_col) not in ('', None) and int(float(r['epoch'])) < job['end_epoch']
        ]
        if not values:
            continue
        final_value, final_epoch = values[-1]
        best_value, best_epoch = (max if maximize else min)(values, key=lambda v: (v[0], -v[1]))  # ties: earlier
        trial = trials[job['trial']]
        row = OrderedDict(trial=job['trial'], job=job['id'].split('/', 1)[1], kind=job['kind'], repeat=trial['repeat'])
        row.update(trial['params'])
        wsd = job.get('wsd') or {}
        row.update(point=wsd.get('point', ''), decay_epochs=wsd.get('decay_epochs', ''),
                   decay_fn=wsd.get('decay_fn', ''))
        row.update({
            'epochs': final_epoch + 1,
            metric_col: final_value,
            f'best_{metric_col}': best_value,
            'best_epoch': best_epoch,
            'status': executor.states[job['id']].status,
            'config_hash': job['config_hash'],
            'dir': job['dir'],
        })
        results.append(row)
    if results:
        results.sort(key=lambda r: r[metric_col], reverse=metric_maximize(plan, metric_col))
    return results, metric_col or metric_name


def aggregate_repeats(results: List[Dict[str, Any]], metric_col: str, space_keys: List[str]) -> List[Dict[str, Any]]:
    """Mean / std of the final metric over repeats of the same parameters and job."""
    groups: Dict[Tuple, List[Dict[str, Any]]] = OrderedDict()
    for r in results:
        if r['status'] != 'completed':
            continue
        key = tuple(json.dumps(r.get(k), sort_keys=True, default=str) for k in space_keys if k != 'seed') + (r['job'],)
        groups.setdefault(key, []).append(r)
    rows = []
    for key, rs in groups.items():
        values = [r[metric_col] for r in rs]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.
        row = OrderedDict((k, rs[0].get(k)) for k in space_keys if k != 'seed')
        row.update(job=rs[0]['job'], n=len(values))
        row[f'mean_{metric_col}'] = round(mean, 4)
        row[f'std_{metric_col}'] = round(std, 4)
        rows.append(row)
    return rows


def show_results(plan: Dict[str, Any], top: int = 20, csv_path: Optional[str] = None):
    results, metric_col = collect_results(plan)
    if not results:
        print('No results yet.')
        return
    csv_path = csv_path or os.path.join(plan['root'], 'results.csv')
    columns = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in columns})
    show = ['trial', 'job'] + list(plan['space_keys'])
    if plan.get('wsd'):
        show += ['point', 'decay_epochs', 'decay_fn']
    show += ['epochs', metric_col, f'best_{metric_col}', 'best_epoch', 'status']
    for r in results:
        for k in (metric_col, f'best_{metric_col}'):
            r[k] = round(r[k], 4)
    direction = 'maximize' if metric_maximize(plan, metric_col) else 'minimize'
    print(f'Top {min(top, len(results))} of {len(results)} runs by final {metric_col} '
          f'({direction}), budgets differ so compare like with like:')
    print_table(results[:top], show)
    repeats = (plan.get('search') or {}).get('repeats') or 1
    if repeats > 1:
        agg = aggregate_repeats(results, metric_col, plan['space_keys'])
        agg.sort(key=lambda r: r[f'mean_{metric_col}'], reverse=metric_maximize(plan, metric_col))
        print(f'\nOver {repeats} repeats:')
        print_table(agg[:top], list(agg[0].keys()) if agg else [])
    print(f'\nWrote {csv_path}')


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------

def load_spec(path: str) -> Dict[str, Any]:
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    if not isinstance(spec, dict):
        raise SweepError(f'{path} must be a mapping.')
    if not spec.get('name'):
        spec['name'] = os.path.splitext(os.path.basename(path))[0]
    return spec


def spec_from_cli(args: argparse.Namespace, train_argv: List[str]) -> Dict[str, Any]:
    """Build the same spec a YAML file would give from --sweep axes and trailing train.py arguments."""
    space: Dict[str, Any] = {}
    for item in args.sweep or []:
        if '=' not in item:
            raise SweepError(f'--sweep expects NAME=V1,V2,..., got "{item}".')
        key, values = item.split('=', 1)
        key = key.strip().replace('-', '_')
        space[key] = [parse_cli_value(v.strip()) for v in values.split(',') if v.strip()]
    train_block: Dict[str, Any] = {}
    if train_argv:
        # only the arguments the user actually gave, the planner fills in the parser defaults
        given = train_script.parser.parse_args(train_argv)
        defaults = train_script.parser.parse_args([])
        train_block = {k: v for k, v in vars(given).items() if v != getattr(defaults, k)}
    search: Dict[str, Any] = {'mode': 'random' if args.random else 'grid', 'seed': args.seed, 'space': space}
    if args.random:
        search['n_trials'] = args.random
    if args.repeats > 1:
        search['repeats'] = args.repeats
    spec: Dict[str, Any] = {
        'name': args.name,
        'output': args.output,
        'train': train_block,
        'search': search,
        'execution': {'gpus_per_job': args.gpus_per_job},
    }
    if args.metric:
        spec['metric'] = {'name': args.metric}
    if args.wsd_points:
        spec['wsd'] = {'points': args.wsd_points}
        if args.wsd_decay_epochs:
            spec['wsd']['decay_epochs'] = args.wsd_decay_epochs
        if args.wsd_decay_fn:
            spec['wsd']['decay_fn'] = args.wsd_decay_fn
    if not spec['name']:
        digest = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:8]
        spec['name'] = f'sweep-{digest}'
    return spec


def resolve_target(args: argparse.Namespace, train_argv: List[str]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Return (plan, spec). A plan comes from a sweep dir, or is built from a spec file / CLI axes."""
    target = args.target
    if target and os.path.isdir(target):
        return load_plan(target), None
    if target:
        if not os.path.isfile(target):
            raise SweepError(f'{target} is neither a sweep directory nor a spec file.')
        spec = load_spec(target)
        spec_dir = os.path.dirname(os.path.abspath(target))
    else:
        if not getattr(args, 'sweep', None) and not getattr(args, 'wsd_points', None):
            raise SweepError('give a sweep directory, a spec YAML, or --sweep NAME=V1,V2 axes.')
        spec = spec_from_cli(args, train_argv)
        spec_dir = os.getcwd()
    return plan_sweep(spec, spec_dir), spec


def parse_gpus(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    gpus = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            gpus.extend(range(int(lo), int(hi) + 1))
        else:
            gpus.append(int(part))
    return gpus


def add_spec_arguments(p: argparse.ArgumentParser):
    p.add_argument('target', nargs='?', help='sweep directory (with plan.json) or sweep spec YAML')
    p.add_argument('--name', default=None, help='sweep name (CLI sweeps, default: derived from the spec)')
    p.add_argument('--output', default=os.path.join('output', 'sweeps'), metavar='DIR',
                   help='parent directory for CLI sweeps, the sweep lives in DIR/NAME')
    p.add_argument('--sweep', action='append', metavar='NAME=V1,V2,...',
                   help='search axis over a train.py argument, repeatable, values parsed as YAML')
    p.add_argument('--random', type=int, default=0, metavar='N', help='random search with N trials instead of a grid')
    p.add_argument('--seed', type=int, default=42, help='random search seed')
    p.add_argument('--repeats', type=int, default=1, help='repeat each trial with seeds base, base+1, ...')
    p.add_argument('--metric', default=None, help='summary.csv eval metric to rank by (default: train eval metric)')
    p.add_argument('--gpus-per-job', type=int, default=1, help='GPUs per job, >1 launches via torchrun')
    p.add_argument('--wsd-points', type=int, nargs='+', default=None, metavar='N', help='WSD branch points')
    p.add_argument('--wsd-decay-epochs', type=int, nargs='+', default=None, metavar='N')
    p.add_argument('--wsd-decay-fn', nargs='+', default=None, metavar='FN')
    p.add_argument('--force', action='store_true', help='overwrite an existing, different plan in the sweep dir')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    argv = sys.argv[1:]
    train_argv: List[str] = []
    if '--' in argv:
        i = argv.index('--')
        argv, train_argv = argv[:i], argv[i + 1:]

    parser = argparse.ArgumentParser(description='timm hyperparameter sweep planner and runner')
    sub = parser.add_subparsers(dest='command', required=True)
    p_plan = sub.add_parser('plan', help='expand a sweep into per-job configs and a plan without running anything')
    add_spec_arguments(p_plan)
    p_run = sub.add_parser('run', help='plan if needed, then run ready jobs on local GPUs until nothing is left')
    add_spec_arguments(p_run)
    p_run.add_argument('--gpus', default=None, metavar='IDS', help='GPU ids to use, e.g. 0,1,2,3 or 0-7')
    p_run.add_argument('--max-parallel', type=int, default=None, help='concurrent jobs (default: one per GPU, else 1)')
    p_run.add_argument('--retries', type=int, default=0, help='automatic retries of a failed job in this run')
    p_run.add_argument('--poll', type=float, default=10., metavar='SEC', help='poll interval')
    p_run.add_argument('--dry-run', action='store_true', help='print the launch order without running')
    p_run.add_argument('--train-args', default='', metavar='ARGS', help='extra args appended to every train.py call')
    p_run.add_argument('--quiet', action='store_true',
                       help='do not echo job output to the console, it is always written to each job\'s run.log')
    p_status = sub.add_parser('status', help='show job states and progress')
    p_status.add_argument('target')
    p_results = sub.add_parser('results', help='collect summary.csv metrics into results.csv and print a ranking')
    p_results.add_argument('target')
    p_results.add_argument('--top', type=int, default=20)
    p_results.add_argument('--csv', default=None, help='results file (default: <sweep>/results.csv)')
    args = parser.parse_args(argv)

    try:
        if args.command == 'plan':
            plan, spec = resolve_target(args, train_argv)
            if spec is None:
                raise SweepError('plan needs a spec file or --sweep axes, that target is already a planned sweep.')
            root = write_plan(plan, spec, force=args.force)
            counts: Dict[str, int] = {}
            for j in plan['jobs']:
                counts[j['kind']] = counts.get(j['kind'], 0) + 1
            print(f'Planned {len(plan["trials"])} trials, {len(plan["jobs"])} jobs '
                  f'({", ".join(f"{v} {k}" for k, v in counts.items())}) in {root}')
            print(f'Run with: python sweep.py run {root} --gpus 0')
        elif args.command == 'run':
            plan, spec = resolve_target(args, train_argv)
            if spec is not None:
                write_plan(plan, spec, force=args.force)
                plan = load_plan(plan['root'])
            executor = Executor(
                plan, gpus=parse_gpus(args.gpus), max_parallel=args.max_parallel, retries=args.retries,
                poll_interval=args.poll, dry_run=args.dry_run, extra_args=args.train_args.split(),
                echo=not args.quiet,
            )
            counts = executor.run()
            print('Jobs: ' + ', '.join(f'{k}: {v}' for k, v in sorted(counts.items())))
            all_done = set(counts) == {'completed'}
            if all_done and not args.dry_run:
                show_results(plan)
            if not all_done and not args.dry_run:
                sys.exit(1)  # something failed, is blocked, or is still running / pending elsewhere
        elif args.command == 'status':
            show_status(load_plan(args.target))
        elif args.command == 'results':
            show_results(load_plan(args.target), top=args.top, csv_path=args.csv)
    except SweepError as e:
        _logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
