#!/usr/bin/env python3
""" Bulk Model Script Runner

Run validation or benchmark script in separate process for each model

Benchmark all 'vit*' models:
python bulk_runner.py  --model-list 'vit*' --results-file vit_bench.csv benchmark.py --amp -b 512

Validate all models:
python bulk_runner.py  --model-list all --results-file val.csv --pretrained validate.py /imagenet/validation/ --amp -b 512 --retry

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import sys
import csv
import json
import subprocess
import time
from typing import Callable, List, Tuple, Union


from timm.models import is_model, list_models, get_pretrained_cfg, get_arch_pretrained_cfgs


parser = argparse.ArgumentParser(description='Per-model process launcher')

# model and results args
parser.add_argument(
    '--model-list', metavar='NAME', default='',
    help='txt file based list of model names to benchmark')
parser.add_argument(
    '--results-file', default='', type=str, metavar='FILENAME',
    help='Output csv file for validation results (summary)')
parser.add_argument(
    '--sort-key', default='', type=str, metavar='COL',
    help='Specify sort key for results csv')
parser.add_argument(
    "--pretrained", action='store_true',
    help="only run models with pretrained weights")

parser.add_argument(
    "--delay",
    type=float,
    default=0,
    help="Interval, in seconds, to delay between model invocations.",
)
parser.add_argument(
    "--start_method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"],
    help="Multiprocessing start method to use when creating workers.",
)
parser.add_argument(
    "--no_python",
    help="Skip prepending the script with 'python' - just execute it directly. Useful "
         "when the script is not a Python script.",
)
parser.add_argument(
    "-m",
    "--module",
    help="Change each process to interpret the launch script as a Python module, executing "
         "with the same behavior as 'python -m'.",
)

# positional
parser.add_argument(
    "script", type=str,
    help="Full path to the program/script to be launched for each model config.",
)
parser.add_argument("script_args", nargs=argparse.REMAINDER)


def cmd_from_args(args) -> Tuple[Union[Callable, str], List[str]]:
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    with_python = not args.no_python
    cmd: Union[Callable, str]
    cmd_args = []
    if with_python:
        cmd = os.getenv("PYTHON_EXEC", sys.executable)
        cmd_args.append("-u")
        if args.module:
            cmd_args.append("-m")
        cmd_args.append(args.script)
    else:
        if args.module:
            raise ValueError(
                "Don't use both the '--no_python' flag"
                " and the '--module' flag at the same time."
            )
        cmd = args.script
    cmd_args.extend(args.script_args)

    return cmd, cmd_args


def _get_model_cfgs(
        model_names,
        num_classes=None,
        expand_train_test=False,
        include_crop=True,
        expand_arch=False,
):
    model_cfgs = set()

    for name in model_names:
        if expand_arch:
            pt_cfgs = get_arch_pretrained_cfgs(name).values()
        else:
            pt_cfg = get_pretrained_cfg(name)
            pt_cfgs = [pt_cfg] if pt_cfg is not None else []

        for cfg in pt_cfgs:
            if cfg.input_size is None:
                continue
            if num_classes is not None and getattr(cfg, 'num_classes', 0) != num_classes:
                continue

            # Add main configuration
            size = cfg.input_size[-1]
            if include_crop:
                model_cfgs.add((name, size, cfg.crop_pct))
            else:
                model_cfgs.add((name, size))

            # Add test configuration if required
            if expand_train_test and cfg.test_input_size is not None:
                test_size = cfg.test_input_size[-1]
                if include_crop:
                    test_crop = cfg.test_crop_pct or cfg.crop_pct
                    model_cfgs.add((name, test_size, test_crop))
                else:
                    model_cfgs.add((name, test_size))

    # Format the output
    if include_crop:
        return [(n, {'img-size': r, 'crop-pct': cp}) for n, r, cp in sorted(model_cfgs)]
    else:
        return [(n, {'img-size': r}) for n, r in sorted(model_cfgs)]


def main():
    args = parser.parse_args()
    cmd, cmd_args = cmd_from_args(args)

    model_cfgs = []
    if args.model_list == 'all':
        model_names = list_models(
            pretrained=args.pretrained,  # only include models w/ pretrained checkpoints if set
        )
        model_cfgs = [(n, None) for n in model_names]
    elif args.model_list == 'all_in1k':
        model_names = list_models(pretrained=True)
        model_cfgs = _get_model_cfgs(model_names, num_classes=1000, expand_train_test=True)
    elif args.model_list == 'all_res':
        model_names = list_models()
        model_cfgs = _get_model_cfgs(model_names, expand_train_test=True, include_crop=False, expand_arch=True)
    elif not is_model(args.model_list):
        # model name doesn't exist, try as wildcard filter
        model_names = list_models(args.model_list)
        model_cfgs = [(n, None) for n in model_names]

    if not model_cfgs and os.path.exists(args.model_list):
        with open(args.model_list) as f:
            model_names = [line.rstrip() for line in f]
            model_cfgs = _get_model_cfgs(
                model_names,
                #num_classes=1000,
                expand_train_test=True,
                #include_crop=False,
            )

    if len(model_cfgs):
        results_file = args.results_file or './results.csv'
        results = []
        errors = []
        model_strings = '\n'.join([f'{x[0]}, {x[1]}' for x in model_cfgs])
        print(f"Running script on these models:\n {model_strings}")
        if not args.sort_key:
            if 'benchmark' in args.script:
                if any(['train' in a for a in args.script_args]):
                    sort_key = 'train_samples_per_sec'
                else:
                    sort_key = 'infer_samples_per_sec'
            else:
                sort_key = 'top1'
        else:
            sort_key = args.sort_key
        print(f'Script: {args.script}, Args: {args.script_args}, Sort key: {sort_key}')

        try:
            for m, ax in model_cfgs:
                if not m:
                    continue
                args_str = (cmd, *[str(e) for e in cmd_args], '--model', m)
                if ax is not None:
                    extra_args = [(f'--{k}', str(v)) for k, v in ax.items()]
                    extra_args = [i for t in extra_args for i in t]
                    args_str += tuple(extra_args)
                try:
                    o = subprocess.check_output(args=args_str).decode('utf-8').split('--result')[-1]
                    r = json.loads(o)
                    results.append(r)
                except Exception as e:
                    # FIXME batch_size retry loop is currently done in either validation.py or benchmark.py
                    # for further robustness (but more overhead), we may want to manage that by looping here...
                    errors.append(dict(model=m, error=str(e)))
                if args.delay:
                    time.sleep(args.delay)
        except KeyboardInterrupt as e:
            pass

        errors.extend(list(filter(lambda x: 'error' in x, results)))
        if errors:
            print(f'{len(errors)} models had errors during run.')
            for e in errors:
                if 'model' in e:
                    print(f"\t {e['model']} ({e.get('error', 'Unknown')})")
                else:
                    print(e)

        results = list(filter(lambda x: 'error' not in x, results))

        no_sortkey = list(filter(lambda x: sort_key not in x, results))
        if no_sortkey:
            print(f'{len(no_sortkey)} results missing sort key, skipping sort.')
        else:
            results = sorted(results, key=lambda x: x[sort_key], reverse=True)

        if len(results):
            print(f'{len(results)} models run successfully. Saving results to {results_file}.')
            write_results(results_file, results)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
