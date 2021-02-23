#!/usr/bin/env python3
""" Model Benchmark Script

An inference and train step benchmark script for timm models.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress
from functools import partial

from timm.models import create_model, is_model, list_models
from timm.optim import create_optimizer
from timm.data import resolve_data_config
from timm.utils import AverageMeter, setup_default_logging


has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch Benchmark')

# benchmark specific args
parser.add_argument('--bench', default='both', type=str,
                    help="Benchmark mode. One of 'inference', 'train', 'both'. Defaults to 'inference'")
parser.add_argument('--detail', action='store_true', default=False,
                    help='Provide train fwd/bwd/opt breakdown detail if True. Defaults to False')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')

# common inference / train args
parser.add_argument('--model', '-m', metavar='NAME', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')


# train optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# model regularization / loss params that impact model or loss fn
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')


def timestamp(sync=False):
    return time.perf_counter()


def cuda_timestamp(sync=False, device=None):
    if sync:
        torch.cuda.synchronize(device=device)
    return time.perf_counter()


def count_params(model):
    return sum([m.numel() for m in model.parameters()])


class BenchmarkRunner:
    def __init__(self, model_name, detail=False, device='cuda', torchscript=False, **kwargs):
        self.model_name = model_name
        self.detail = detail
        self.device = device
        self.model = create_model(
            model_name,
            num_classes=kwargs.pop('num_classes', None),
            in_chans=3,
            global_pool=kwargs.pop('gp', 'fast'),
            scriptable=torchscript).to(device=self.device)
        self.num_classes = self.model.num_classes
        self.param_count = count_params(self.model)
        _logger.info('Model %s created, param count: %d' % (model_name, self.param_count))

        self.channels_last = kwargs.pop('channels_last', False)
        self.use_amp = kwargs.pop('use_amp', '')
        self.amp_autocast = torch.cuda.amp.autocast if self.use_amp == 'native' else suppress
        if torchscript:
            self.model = torch.jit.script(self.model)

        data_config = resolve_data_config(kwargs, model=self.model, use_test_size=True)
        self.input_size = data_config['input_size']
        self.batch_size = kwargs.pop('batch_size', 256)

        self.example_inputs = None
        self.num_warm_iter = 10
        self.num_bench_iter = 50
        self.log_freq = 10
        if 'cuda' in self.device:
            self.time_fn = partial(cuda_timestamp, device=self.device)
        else:
            self.time_fn = timestamp

    def _init_input(self):
        self.example_inputs = torch.randn((self.batch_size,) + self.input_size, device=self.device)
        if self.channels_last:
            self.example_inputs = self.example_inputs.contiguous(memory_format=torch.channels_last)


class InferenceBenchmarkRunner(BenchmarkRunner):

    def __init__(self, model_name, device='cuda', torchscript=False, **kwargs):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwargs)
        self.model.eval()
        if self.use_amp == 'apex':
            self.model = amp.initialize(self.model, opt_level='O1')
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

    def run(self):
        def _step():
            t_step_start = self.time_fn()
            with self.amp_autocast():
                output = self.model(self.example_inputs)
            t_step_end = self.time_fn(True)
            return t_step_end - t_step_start

        _logger.info(
            f'Running inference benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

        with torch.no_grad():
            self._init_input()

            for _ in range(self.num_warm_iter):
                _step()

            total_step = 0.
            num_samples = 0
            t_run_start = self.time_fn()
            for i in range(self.num_bench_iter):
                delta_fwd = _step()
                total_step += delta_fwd
                num_samples += self.batch_size
                if (i + 1) % self.log_freq == 0:
                    _logger.info(
                        f"Infer [{i + 1}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_step / num_samples:0.3f} ms/sample.")
            t_run_end = self.time_fn(True)
            t_run_elapsed = t_run_end - t_run_start

        results = dict(
            samples_per_sec=round(num_samples / t_run_elapsed, 2),
            step_time=round(1000 * total_step / num_samples, 3),
            batch_size=self.batch_size,
            param_count=round(self.param_count / 1e6, 2),
        )

        _logger.info(
            f"Inference benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/sample")

        return results


class TrainBenchmarkRunner(BenchmarkRunner):

    def __init__(self, model_name, device='cuda', torchscript=False, **kwargs):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwargs)
        self.model.train()

        if kwargs.pop('smoothing', 0) > 0:
            self.loss = nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss = nn.CrossEntropyLoss().to(self.device)
        self.target_shape = tuple()

        self.optimizer = create_optimizer(
            self.model,
            opt_name=kwargs.pop('opt', 'sgd'),
            lr=kwargs.pop('lr', 1e-4))

        if self.use_amp == 'apex':
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

    def _gen_target(self, batch_size):
        return torch.empty(
            (batch_size,) + self.target_shape, device=self.device, dtype=torch.long).random_(self.num_classes)

    def run(self):
        def _step(detail=False):
            self.optimizer.zero_grad()  # can this be ignored?
            t_start = self.time_fn()
            t_fwd_end = t_start
            t_bwd_end = t_start
            with self.amp_autocast():
                output = self.model(self.example_inputs)
                if isinstance(output, tuple):
                    output = output[0]
                if detail:
                    t_fwd_end = self.time_fn(True)
                target = self._gen_target(output.shape[0])
                self.loss(output, target).backward()
                if detail:
                    t_bwd_end = self.time_fn(True)
            self.optimizer.step()
            t_end = self.time_fn(True)
            if detail:
                delta_fwd = t_fwd_end - t_start
                delta_bwd = t_bwd_end - t_fwd_end
                delta_opt = t_end - t_bwd_end
                return delta_fwd, delta_bwd, delta_opt
            else:
                delta_step = t_end - t_start
                return delta_step

        _logger.info(
            f'Running train benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

        self._init_input()

        for _ in range(self.num_warm_iter):
            _step()

        t_run_start = self.time_fn()
        if self.detail:
            total_fwd = 0.
            total_bwd = 0.
            total_opt = 0.
            num_samples = 0
            for i in range(self.num_bench_iter):
                delta_fwd, delta_bwd, delta_opt = _step(True)
                num_samples += self.batch_size
                total_fwd += delta_fwd
                total_bwd += delta_bwd
                total_opt += delta_opt
                if (i + 1) % self.log_freq == 0:
                    total_step = total_fwd + total_bwd + total_opt
                    _logger.info(
                        f"Train [{i + 1}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_fwd / num_samples:0.3f} ms/sample fwd,"
                        f" {1000 * total_bwd / num_samples:0.3f} ms/sample bwd,"
                        f" {1000 * total_opt / num_samples:0.3f} ms/sample opt."
                    )
            total_step = total_fwd + total_bwd + total_opt
            t_run_elapsed = self.time_fn() - t_run_start
            results = dict(
                samples_per_sec=round(num_samples / t_run_elapsed, 2),
                step_time=round(1000 * total_step / num_samples, 3),
                fwd_time=round(1000 * total_fwd / num_samples, 3),
                bwd_time=round(1000 * total_bwd / num_samples, 3),
                opt_time=round(1000 * total_opt / num_samples, 3),
                batch_size=self.batch_size,
                param_count=round(self.param_count / 1e6, 2),
            )
        else:
            total_step = 0.
            num_samples = 0
            for i in range(self.num_bench_iter):
                delta_step = _step(False)
                num_samples += self.batch_size
                total_step += delta_step
                if (i + 1) % self.log_freq == 0:
                    _logger.info(
                        f"Train [{i + 1}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_step / num_samples:0.3f} ms/sample.")
            t_run_elapsed = self.time_fn() - t_run_start
            results = dict(
                samples_per_sec=round(num_samples / t_run_elapsed, 2),
                step_time=round(1000 * total_step / num_samples, 3),
                batch_size=self.batch_size,
                param_count=round(self.param_count / 1e6, 2),
            )

        _logger.info(
            f"Train benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/sample")

        return results


def decay_batch_exp(batch_size, factor=0.5, divisor=16):
    out_batch_size = batch_size * factor
    if out_batch_size > divisor:
        out_batch_size = (out_batch_size + 1) // divisor * divisor
    else:
        out_batch_size = batch_size - 1
    return max(0, int(out_batch_size))


def _try_run(model_name, bench_fn, initial_batch_size, bench_kwargs):
    batch_size = initial_batch_size
    results = dict()
    while batch_size >= 1:
        try:
            bench = bench_fn(model_name=model_name, batch_size=batch_size, **bench_kwargs)
            results = bench.run()
            return results
        except RuntimeError as e:
            torch.cuda.empty_cache()
            batch_size = decay_batch_exp(batch_size)
            print(f'Reducing batch size to {batch_size}')
    return results


def benchmark(args):
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    if args.native_amp:
        args.use_amp = 'native'
        _logger.info('Benchmarking in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        args.use_amp = 'apex'
        _logger.info('Benchmarking in mixed precision with NVIDIA APEX AMP.')
    else:
        args.use_amp = ''
        _logger.info('Benchmarking in float32. AMP not enabled.')

    bench_kwargs = vars(args).copy()
    model = bench_kwargs.pop('model')
    batch_size = bench_kwargs.pop('batch_size')

    bench_fns = (InferenceBenchmarkRunner,)
    prefixes = ('infer',)
    if args.bench == 'both':
        bench_fns = (
            InferenceBenchmarkRunner,
            TrainBenchmarkRunner
        )
        prefixes = ('infer', 'train')
    elif args.bench == 'train':
        bench_fns = TrainBenchmarkRunner,
        prefixes = 'train',

    model_results = OrderedDict(model=model)
    for prefix, bench_fn in zip(prefixes, bench_fns):
        run_results = _try_run(model, bench_fn, initial_batch_size=batch_size, bench_kwargs=bench_kwargs)
        if prefix:
            run_results = {'_'.join([prefix, k]): v for k, v in run_results.items()}
        model_results.update(run_results)
    param_count = model_results.pop('infer_param_count', model_results.pop('train_param_count', 0))
    model_results.setdefault('param_count', param_count)
    model_results.pop('train_param_count', 0)
    return model_results


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []

    if args.model == 'all':
        # validate all models in a list of names with pretrained checkpoints
        args.pretrained = True
        model_names = list_models(pretrained=True, exclude_filters=['*in21k'])
        model_cfgs = [(n, None) for n in model_names]
    elif not is_model(args.model):
        # model name doesn't exist, try as wildcard filter
        model_names = list_models(args.model)
        model_cfgs = [(n, None) for n in model_names]

    if len(model_cfgs):
        results_file = args.results_file or './benchmark.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            for m, _ in model_cfgs:
                args.model = m
                r = benchmark(args)
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        sort_key = 'train_samples_per_sec' if 'train' in args.bench else 'infer_samples_per_sec'
        results = sorted(results, key=lambda x: x[sort_key], reverse=True)
        if len(results):
            write_results(results_file, results)

        import json
        json_str = json.dumps(results, indent=4)
        print(json_str)
    else:
        benchmark(args)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
