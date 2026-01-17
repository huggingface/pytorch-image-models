#!/usr/bin/env python3
"""Hyperparameter sweep runner for timm training.

Supports grid search and random search with subprocess parallelization.
Each trial runs in an isolated process with configurable GPU assignment.

CLI usage::

    # As a module
    python -m timm.apps.sweep sweeps/lr_sweep.yaml

    # As installed script
    timm-sweep sweeps/lr_sweep.yaml

    # Resume incomplete sweep
    timm-sweep sweeps/lr_sweep.yaml --resume

    # Custom training script
    timm-sweep sweeps/lr_sweep.yaml --train-script timm.apps.train_ssl

Sweep config format (YAML)::

    base_config:
      path: configs/resnet50.yaml  # or inline config

    sweep:
      method: grid  # or 'random'
      num_trials: 20  # for random search
      metric:
        name: eval_top1
        goal: maximize
      parameters:
        optimizer.lr:
          values: [0.01, 0.05, 0.1]
        optimizer.weight_decay:
          values: [1e-4, 1e-5]

    execution:
      parallel: subprocess
      max_workers: 4
      gpus: [0, 1, 2, 3]
      output_dir: ./output/sweeps

Programmatic usage::

    from timm.apps.sweep import run_sweep

    results = run_sweep('sweeps/lr_sweep.yaml', resume=False)
    for r in results:
        print(f"Trial {r.trial_id}: {r.status}, metrics={r.metrics}")
"""
import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

_logger = logging.getLogger('sweep')


@dataclass
class SweepResult:
    """Result from a single sweep trial."""
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    output_dir: str
    status: str  # 'completed', 'failed', 'running'
    error: Optional[str] = None


@dataclass
class SweepState:
    """Persistent state for sweep resumption."""
    sweep_id: str
    config_path: str
    completed_trials: List[int] = field(default_factory=list)
    results: List[Dict] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Save sweep state to file."""
        with open(path, 'w') as f:
            json.dump({
                'sweep_id': self.sweep_id,
                'config_path': self.config_path,
                'completed_trials': self.completed_trials,
                'results': self.results,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SweepState':
        """Load sweep state from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        state = cls(
            sweep_id=data['sweep_id'],
            config_path=data['config_path'],
            completed_trials=data['completed_trials'],
        )
        state.results = data['results']
        return state


def load_sweep_config(config_path: str) -> dict:
    """Load sweep configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_grid_trials(parameters: Dict[str, dict]) -> Iterator[Dict[str, Any]]:
    """Generate all parameter combinations for grid search.

    Args:
        parameters: Dict mapping param names to {'values': [...]}.

    Yields:
        Dict mapping param names to values for each combination.
    """
    param_names = list(parameters.keys())
    param_values = [parameters[name]['values'] for name in param_names]

    for combination in itertools.product(*param_values):
        yield dict(zip(param_names, combination))


def generate_random_trials(
    parameters: Dict[str, dict],
    num_trials: int,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """Generate random parameter combinations.

    Args:
        parameters: Dict mapping param names to distribution specs.
        num_trials: Number of trials to generate.
        seed: Random seed for reproducibility.

    Yields:
        Dict mapping param names to sampled values.

    Supported distributions:
        - values: [list] - uniform choice from list
        - distribution: uniform, min: float, max: float
        - distribution: log_uniform, min: float, max: float
        - distribution: int_uniform, min: int, max: int
    """
    rng = random.Random(seed)

    for _ in range(num_trials):
        trial_params = {}
        for name, param_cfg in parameters.items():
            if 'values' in param_cfg:
                trial_params[name] = rng.choice(param_cfg['values'])
            elif 'distribution' in param_cfg:
                dist = param_cfg['distribution']
                if dist == 'uniform':
                    trial_params[name] = rng.uniform(param_cfg['min'], param_cfg['max'])
                elif dist == 'log_uniform':
                    log_min = math.log(param_cfg['min'])
                    log_max = math.log(param_cfg['max'])
                    trial_params[name] = math.exp(rng.uniform(log_min, log_max))
                elif dist == 'int_uniform':
                    trial_params[name] = rng.randint(param_cfg['min'], param_cfg['max'])
                elif dist == 'choice':
                    trial_params[name] = rng.choice(param_cfg['options'])
                else:
                    raise ValueError(f"Unknown distribution: {dist}")
            else:
                raise ValueError(f"Parameter {name} must have 'values' or 'distribution'")
        yield trial_params


def create_trial_config(
    base_config: dict,
    trial_params: Dict[str, Any],
) -> dict:
    """Merge trial parameters into base config.

    Parameters use dot notation (e.g., 'optimizer.lr' -> config['optimizer']['lr']).

    Args:
        base_config: Base configuration dict.
        trial_params: Trial-specific parameter overrides.

    Returns:
        Merged configuration dict.
    """
    import copy
    config = copy.deepcopy(base_config)

    for param_path, value in trial_params.items():
        keys = param_path.split('.')
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value

    return config


def run_trial(
    trial_id: int,
    trial_params: Dict[str, Any],
    base_config: dict,
    output_dir: str,
    gpu_id: int,
    train_script: str = 'timm.apps.train_cls',
) -> SweepResult:
    """Run a single training trial as subprocess.

    Args:
        trial_id: Unique trial identifier.
        trial_params: Parameter values for this trial.
        base_config: Base training config.
        output_dir: Sweep output directory.
        gpu_id: GPU to use for this trial.
        train_script: Training script/module to run.

    Returns:
        SweepResult with trial outcome.
    """
    trial_output_dir = os.path.join(output_dir, f'trial_{trial_id:04d}')
    os.makedirs(trial_output_dir, exist_ok=True)

    # Create merged config
    trial_config = create_trial_config(base_config, trial_params)

    # Save trial config
    config_path = os.path.join(trial_output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(trial_config, f, default_flow_style=False)

    # Save trial params for reference
    params_path = os.path.join(trial_output_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(trial_params, f, indent=2)

    # Build command
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Support both module paths (timm.apps.train_cls) and script paths (train_cls.py)
    if '.' in train_script and not train_script.endswith('.py'):
        # Module path - run with python -m
        cmd = [
            sys.executable, '-m', train_script,
            '-c', config_path,
            '--misc.output', trial_output_dir,
            '--misc.experiment', f'trial_{trial_id:04d}',
        ]
    else:
        # Script path
        cmd = [
            sys.executable, train_script,
            '-c', config_path,
            '--misc.output', trial_output_dir,
            '--misc.experiment', f'trial_{trial_id:04d}',
        ]

    _logger.info(f'Starting trial {trial_id} on GPU {gpu_id}')

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=86400,  # 24 hour timeout
        )

        if result.returncode != 0:
            return SweepResult(
                trial_id=trial_id,
                params=trial_params,
                metrics={},
                output_dir=trial_output_dir,
                status='failed',
                error=result.stderr[-2000:] if result.stderr else 'Unknown error',
            )

        # Parse results from summary.csv
        metrics = parse_trial_results(trial_output_dir)

        return SweepResult(
            trial_id=trial_id,
            params=trial_params,
            metrics=metrics,
            output_dir=trial_output_dir,
            status='completed',
        )

    except subprocess.TimeoutExpired:
        return SweepResult(
            trial_id=trial_id,
            params=trial_params,
            metrics={},
            output_dir=trial_output_dir,
            status='failed',
            error='Timeout (24h)',
        )
    except Exception as e:
        return SweepResult(
            trial_id=trial_id,
            params=trial_params,
            metrics={},
            output_dir=trial_output_dir,
            status='failed',
            error=str(e),
        )


def parse_trial_results(output_dir: str) -> Dict[str, float]:
    """Parse final metrics from trial output.

    Reads the summary.csv file and returns metrics from the last row.

    Args:
        output_dir: Trial output directory.

    Returns:
        Dict of metric names to values.
    """
    # Find summary.csv (might be in experiment subdirectory)
    summary_paths = [
        os.path.join(output_dir, 'summary.csv'),
    ]

    # Also check subdirectories
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            summary_paths.append(os.path.join(subdir_path, 'summary.csv'))

    for summary_path in summary_paths:
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                if not rows:
                    continue

                # Return last row metrics
                last_row = rows[-1]
                metrics = {}
                for key, value in last_row.items():
                    try:
                        metrics[key] = float(value)
                    except (ValueError, TypeError):
                        pass
                return metrics
            except Exception:
                continue

    return {}


def generate_sweep_report(
    state: SweepState,
    output_dir: str,
    metric_cfg: dict,
) -> None:
    """Generate sweep summary report.

    Args:
        state: Sweep state with results.
        output_dir: Sweep output directory.
        metric_cfg: Metric configuration with 'name' and 'goal'.
    """
    metric_name = metric_cfg['name']
    goal = metric_cfg.get('goal', 'maximize')

    # Filter completed trials with target metric
    completed = [
        r for r in state.results
        if r.get('status') == 'completed' and metric_name in r.get('metrics', {})
    ]

    # Sort by target metric
    completed.sort(
        key=lambda r: r['metrics'][metric_name],
        reverse=(goal == 'maximize'),
    )

    report = {
        'sweep_id': state.sweep_id,
        'total_trials': len(state.results),
        'completed_trials': len(completed),
        'failed_trials': len([r for r in state.results if r.get('status') == 'failed']),
        'metric': metric_name,
        'goal': goal,
        'best_trial': None,
        'top_results': [],
    }

    if completed:
        best = completed[0]
        report['best_trial'] = {
            'trial_id': best['trial_id'],
            'params': best['params'],
            'metrics': best['metrics'],
            'output_dir': best['output_dir'],
        }

    # Top 20 results
    for r in completed[:20]:
        report['top_results'].append({
            'trial_id': r['trial_id'],
            'params': r['params'],
            'metrics': r['metrics'],
        })

    report_path = os.path.join(output_dir, 'sweep_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    _logger.info(f'Sweep report saved to {report_path}')
    if report['best_trial']:
        best_value = report['best_trial']['metrics'][metric_name]
        _logger.info(
            f"Best trial: {report['best_trial']['trial_id']} "
            f"with {metric_name}={best_value:.4f}"
        )

    # Also save as CSV for easy analysis
    csv_path = os.path.join(output_dir, 'sweep_results.csv')
    if completed:
        fieldnames = ['trial_id', metric_name] + list(completed[0]['params'].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in completed:
                row = {'trial_id': r['trial_id'], metric_name: r['metrics'][metric_name]}
                row.update(r['params'])
                writer.writerow(row)
        _logger.info(f'Sweep results CSV saved to {csv_path}')


def run_sweep(
    config_path: str,
    resume: bool = False,
    train_script: str = 'timm.apps.train_cls',
) -> List[SweepResult]:
    """Run hyperparameter sweep.

    Args:
        config_path: Path to sweep configuration YAML.
        resume: Whether to resume incomplete sweep.
        train_script: Training script/module to use.

    Returns:
        List of SweepResult for all trials.

    Example::

        from timm.apps.sweep import run_sweep

        results = run_sweep('sweeps/lr_sweep.yaml')
        best = max(results, key=lambda r: r.metrics.get('eval_top1', 0))
        print(f"Best trial: {best.trial_id}, params: {best.params}")
    """
    config = load_sweep_config(config_path)

    sweep_cfg = config['sweep']
    exec_cfg = config.get('execution', {})

    # Setup output directory
    base_output_dir = exec_cfg.get('output_dir', './output/sweeps')
    sweep_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    sweep_output_dir = os.path.join(base_output_dir, sweep_id)

    # Check for resume
    state_path = None
    if resume:
        # Look for existing sweep in base output dir
        existing_sweeps = sorted(Path(base_output_dir).glob('*/sweep_state.json'))
        if existing_sweeps:
            state_path = str(existing_sweeps[-1])
            sweep_output_dir = str(existing_sweeps[-1].parent)
            _logger.info(f'Resuming sweep from {state_path}')

    os.makedirs(sweep_output_dir, exist_ok=True)

    # Load or create sweep state
    if state_path and os.path.exists(state_path):
        state = SweepState.load(state_path)
        _logger.info(f'Resuming with {len(state.completed_trials)} completed trials')
    else:
        state = SweepState(sweep_id=sweep_id, config_path=config_path)
        state_path = os.path.join(sweep_output_dir, 'sweep_state.json')

    # Load base config
    base_config = {}
    if 'base_config' in config:
        if 'path' in config['base_config']:
            with open(config['base_config']['path'], 'r') as f:
                base_config = yaml.safe_load(f)
        else:
            base_config = config['base_config']

    # Generate trials
    method = sweep_cfg.get('method', 'grid')
    parameters = sweep_cfg['parameters']

    if method == 'grid':
        all_trials = list(generate_grid_trials(parameters))
    elif method == 'random':
        num_trials = sweep_cfg.get('num_trials', 20)
        seed = sweep_cfg.get('seed', 42)
        all_trials = list(generate_random_trials(parameters, num_trials, seed))
    else:
        raise ValueError(f"Unknown sweep method: {method}")

    _logger.info(f'Generated {len(all_trials)} trials for {method} search')

    # Filter out completed trials
    pending_trials = [
        (i, t) for i, t in enumerate(all_trials)
        if i not in state.completed_trials
    ]

    _logger.info(f'{len(pending_trials)} trials remaining')

    if not pending_trials:
        _logger.info('All trials complete')
        generate_sweep_report(state, sweep_output_dir, sweep_cfg['metric'])
        return [SweepResult(**r) for r in state.results]

    # Setup GPU pool
    gpus = exec_cfg.get('gpus', [0])
    max_workers = min(
        exec_cfg.get('max_workers', 1),
        len(gpus),
        len(pending_trials),
    )

    parallel_mode = exec_cfg.get('parallel', 'subprocess')

    if parallel_mode == 'sequential' or max_workers == 1:
        # Sequential execution
        for trial_id, trial_params in pending_trials:
            gpu_id = gpus[0]

            result = run_trial(
                trial_id,
                trial_params,
                base_config,
                sweep_output_dir,
                gpu_id,
                train_script,
            )

            state.completed_trials.append(trial_id)
            state.results.append(result.__dict__)
            state.save(state_path)

            status_str = f'{result.status}'
            if result.metrics:
                metric_name = sweep_cfg['metric']['name']
                if metric_name in result.metrics:
                    status_str += f', {metric_name}={result.metrics[metric_name]:.4f}'
            _logger.info(f'Trial {trial_id} finished: {status_str}')
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_trial = {}

            for idx, (trial_id, trial_params) in enumerate(pending_trials):
                gpu_id = gpus[idx % len(gpus)]

                future = executor.submit(
                    run_trial,
                    trial_id,
                    trial_params,
                    base_config,
                    sweep_output_dir,
                    gpu_id,
                    train_script,
                )
                future_to_trial[future] = trial_id

            for future in as_completed(future_to_trial):
                trial_id = future_to_trial[future]
                try:
                    result = future.result()
                    state.completed_trials.append(trial_id)
                    state.results.append(result.__dict__)
                    state.save(state_path)

                    status_str = f'{result.status}'
                    if result.metrics:
                        metric_name = sweep_cfg['metric']['name']
                        if metric_name in result.metrics:
                            status_str += f', {metric_name}={result.metrics[metric_name]:.4f}'
                    _logger.info(f'Trial {trial_id} finished: {status_str}')
                except Exception as e:
                    _logger.error(f'Trial {trial_id} exception: {e}')

    # Generate final report
    generate_sweep_report(state, sweep_output_dir, sweep_cfg['metric'])

    return [SweepResult(**r) for r in state.results]


def main():
    """CLI entrypoint for hyperparameter sweeps."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter sweep runner for timm training',
    )
    parser.add_argument(
        'config',
        type=str,
        help='Sweep configuration YAML file',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume incomplete sweep',
    )
    parser.add_argument(
        '--train-script',
        type=str,
        default='timm.apps.train_cls',
        help='Training script/module to use (default: timm.apps.train_cls)',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    run_sweep(args.config, resume=args.resume, train_script=args.train_script)


if __name__ == '__main__':
    main()
