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
from timm.optim import create_optimizer_v2
from timm.data import resolve_data_config
from timm.utils import setup_default_logging, set_jit_fuser


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

try:
    from deepspeed.profiling.flops_profiler import get_model_profile
    has_deepspeed_profiling = True
except ImportError as e:
    has_deepspeed_profiling = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis
    has_fvcore_profiling = True
except ImportError as e:
    FlopCountAnalysis = None
    has_fvcore_profiling = False


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch Benchmark')

# benchmark specific args
parser.add_argument('--model-list', metavar='NAME', default='',
                    help='txt file based list of model names to benchmark')
parser.add_argument('--bench', default='both', type=str,
                    help="Benchmark mode. One of 'inference', 'train', 'both'. Defaults to 'both'")
parser.add_argument('--detail', action='store_true', default=False,
                    help='Provide train fwd/bwd/opt breakdown detail if True. Defaults to False')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--num-warm-iter', default=10, type=int,
                    metavar='N', help='Number of warmup iterations (default: 10)')
parser.add_argument('--num-bench-iter', default=40, type=int,
                    metavar='N', help='Number of benchmark iterations (default: 40)')

# common inference / train args
parser.add_argument('--model', '-m', metavar='NAME', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='Run inference at train size, not test-input-size if it exists.')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use PyTorch Native AMP for mixed precision training. Overrides --precision arg.')
parser.add_argument('--precision', default='float32', type=str,
                    help='Numeric precision. One of (amp, float32, float16, bfloat16, tf32)')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")


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


def count_params(model: nn.Module):
    return sum([m.numel() for m in model.parameters()])


def resolve_precision(precision: str):
    assert precision in ('amp', 'float16', 'bfloat16', 'float32')
    use_amp = False
    model_dtype = torch.float32
    data_dtype = torch.float32
    if precision == 'amp':
        use_amp = True
    elif precision == 'float16':
        model_dtype = torch.float16
        data_dtype = torch.float16
    elif precision == 'bfloat16':
        model_dtype = torch.bfloat16
        data_dtype = torch.bfloat16
    return use_amp, model_dtype, data_dtype


def profile_deepspeed(model, input_size=(3, 224, 224), batch_size=1, detailed=False):
    macs, _ = get_model_profile(
        model=model,
        input_res=(batch_size,) + input_size,  # input shape or input to the input_constructor
        input_constructor=None,  # if specified, a constructor taking input_res is used as input to the model
        print_profile=detailed,  # prints the model graph with the measured profile attached to each module
        detailed=detailed,  # print the detailed profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return macs, 0  # no activation count in DS


def profile_fvcore(model, input_size=(3, 224, 224), batch_size=1, detailed=False, force_cpu=False):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + input_size, device=device, dtype=dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


class BenchmarkRunner:
    def __init__(
            self, model_name, detail=False, device='cuda', torchscript=False, precision='float32',
            fuser='', num_warm_iter=10, num_bench_iter=50, use_train_size=False, **kwargs):
        self.model_name = model_name
        self.detail = detail
        self.device = device
        self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(precision)
        self.channels_last = kwargs.pop('channels_last', False)
        self.amp_autocast = torch.cuda.amp.autocast if self.use_amp else suppress

        if fuser:
            set_jit_fuser(fuser)
        self.model = create_model(
            model_name,
            num_classes=kwargs.pop('num_classes', None),
            in_chans=3,
            global_pool=kwargs.pop('gp', 'fast'),
            scriptable=torchscript)
        self.model.to(
            device=self.device,
            dtype=self.model_dtype,
            memory_format=torch.channels_last if self.channels_last else None)
        self.num_classes = self.model.num_classes
        self.param_count = count_params(self.model)
        _logger.info('Model %s created, param count: %d' % (model_name, self.param_count))
        self.scripted = False
        if torchscript:
            self.model = torch.jit.script(self.model)
            self.scripted = True

        data_config = resolve_data_config(kwargs, model=self.model, use_test_size=not use_train_size)
        self.input_size = data_config['input_size']
        self.batch_size = kwargs.pop('batch_size', 256)

        self.example_inputs = None
        self.num_warm_iter = num_warm_iter
        self.num_bench_iter = num_bench_iter
        self.log_freq = num_bench_iter // 5
        if 'cuda' in self.device:
            self.time_fn = partial(cuda_timestamp, device=self.device)
        else:
            self.time_fn = timestamp

    def _init_input(self):
        self.example_inputs = torch.randn(
            (self.batch_size,) + self.input_size, device=self.device, dtype=self.data_dtype)
        if self.channels_last:
            self.example_inputs = self.example_inputs.contiguous(memory_format=torch.channels_last)


class InferenceBenchmarkRunner(BenchmarkRunner):

    def __init__(self, model_name, device='cuda', torchscript=False, **kwargs):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwargs)
        self.model.eval()

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
                num_steps = i + 1
                if num_steps % self.log_freq == 0:
                    _logger.info(
                        f"Infer [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_step / num_steps:0.3f} ms/step.")
            t_run_end = self.time_fn(True)
            t_run_elapsed = t_run_end - t_run_start

        results = dict(
            samples_per_sec=round(num_samples / t_run_elapsed, 2),
            step_time=round(1000 * total_step / self.num_bench_iter, 3),
            batch_size=self.batch_size,
            img_size=self.input_size[-1],
            param_count=round(self.param_count / 1e6, 2),
        )

        retries = 0 if self.scripted else 2  # skip profiling if model is scripted
        while retries:
            retries -= 1
            try:
                if has_deepspeed_profiling:
                    macs, _ = profile_deepspeed(self.model, self.input_size)
                    results['gmacs'] = round(macs / 1e9, 2)
                elif has_fvcore_profiling:
                    macs, activations = profile_fvcore(self.model, self.input_size, force_cpu=not retries)
                    results['gmacs'] = round(macs / 1e9, 2)
                    results['macts'] = round(activations / 1e6, 2)
            except RuntimeError as e:
                pass

        _logger.info(
            f"Inference benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step")

        return results


class TrainBenchmarkRunner(BenchmarkRunner):

    def __init__(self, model_name, device='cuda', torchscript=False, **kwargs):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwargs)
        self.model.train()

        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.target_shape = tuple()

        self.optimizer = create_optimizer_v2(
            self.model,
            opt=kwargs.pop('opt', 'sgd'),
            lr=kwargs.pop('lr', 1e-4))

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
                num_steps = (i + 1)
                if num_steps % self.log_freq == 0:
                    total_step = total_fwd + total_bwd + total_opt
                    _logger.info(
                        f"Train [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_fwd / num_steps:0.3f} ms/step fwd,"
                        f" {1000 * total_bwd / num_steps:0.3f} ms/step bwd,"
                        f" {1000 * total_opt / num_steps:0.3f} ms/step opt."
                    )
            total_step = total_fwd + total_bwd + total_opt
            t_run_elapsed = self.time_fn() - t_run_start
            results = dict(
                samples_per_sec=round(num_samples / t_run_elapsed, 2),
                step_time=round(1000 * total_step / self.num_bench_iter, 3),
                fwd_time=round(1000 * total_fwd / self.num_bench_iter, 3),
                bwd_time=round(1000 * total_bwd / self.num_bench_iter, 3),
                opt_time=round(1000 * total_opt / self.num_bench_iter, 3),
                batch_size=self.batch_size,
                img_size=self.input_size[-1],
                param_count=round(self.param_count / 1e6, 2),
            )
        else:
            total_step = 0.
            num_samples = 0
            for i in range(self.num_bench_iter):
                delta_step = _step(False)
                num_samples += self.batch_size
                total_step += delta_step
                num_steps = (i + 1)
                if num_steps % self.log_freq == 0:
                    _logger.info(
                        f"Train [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_step / num_steps:0.3f} ms/step.")
            t_run_elapsed = self.time_fn() - t_run_start
            results = dict(
                samples_per_sec=round(num_samples / t_run_elapsed, 2),
                step_time=round(1000 * total_step / self.num_bench_iter, 3),
                batch_size=self.batch_size,
                img_size=self.input_size[-1],
                param_count=round(self.param_count / 1e6, 2),
            )

        _logger.info(
            f"Train benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/sample")

        return results


class ProfileRunner(BenchmarkRunner):

    def __init__(self, model_name, device='cuda', profiler='', **kwargs):
        super().__init__(model_name=model_name, device=device, **kwargs)
        if not profiler:
            if has_deepspeed_profiling:
                profiler = 'deepspeed'
            elif has_fvcore_profiling:
                profiler = 'fvcore'
        assert profiler, "One of deepspeed or fvcore needs to be installed for profiling to work."
        self.profiler = profiler
        self.model.eval()

    def run(self):
        _logger.info(
            f'Running profiler on {self.model_name} w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

        macs = 0
        activations = 0
        if self.profiler == 'deepspeed':
            macs, _ = profile_deepspeed(self.model, self.input_size, batch_size=self.batch_size, detailed=True)
        elif self.profiler == 'fvcore':
            macs, activations = profile_fvcore(self.model, self.input_size, batch_size=self.batch_size, detailed=True)

        results = dict(
            gmacs=round(macs / 1e9, 2),
            macts=round(activations / 1e6, 2),
            batch_size=self.batch_size,
            img_size=self.input_size[-1],
            param_count=round(self.param_count / 1e6, 2),
        )

        _logger.info(
            f"Profile of {self.model_name} done. "
            f"{results['gmacs']:.2f} GMACs, {results['param_count']:.2f} M params.")

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
    error_str = 'Unknown'
    while batch_size >= 1:
        torch.cuda.empty_cache()
        try:
            bench = bench_fn(model_name=model_name, batch_size=batch_size, **bench_kwargs)
            results = bench.run()
            return results
        except RuntimeError as e:
            error_str = str(e)
            if 'channels_last' in error_str:
                _logger.error(f'{model_name} not supported in channels_last, skipping.')
                break
            _logger.warning(f'"{error_str}" while running benchmark. Reducing batch size to {batch_size} for retry.')
        batch_size = decay_batch_exp(batch_size)
    results['error'] = error_str
    return results


def benchmark(args):
    if args.amp:
        _logger.warning("Overriding precision to 'amp' since --amp flag set.")
        args.precision = 'amp'
    _logger.info(f'Benchmarking in {args.precision} precision. '
                 f'{"NHWC" if args.channels_last else "NCHW"} layout. '
                 f'torchscript {"enabled" if args.torchscript else "disabled"}')

    bench_kwargs = vars(args).copy()
    bench_kwargs.pop('amp')
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
    elif args.bench.startswith('profile'):
        # specific profiler used if included in bench mode string, otherwise default to deepspeed, fallback to fvcore
        if 'deepspeed' in args.bench:
            assert has_deepspeed_profiling, "deepspeed must be installed to use deepspeed flop counter"
            bench_kwargs['profiler'] = 'deepspeed'
        elif 'fvcore' in args.bench:
            assert has_fvcore_profiling, "fvcore must be installed to use fvcore flop counter"
            bench_kwargs['profiler'] = 'fvcore'
        bench_fns = ProfileRunner,
        batch_size = 1

    model_results = OrderedDict(model=model)
    for prefix, bench_fn in zip(prefixes, bench_fns):
        run_results = _try_run(model, bench_fn, initial_batch_size=batch_size, bench_kwargs=bench_kwargs)
        if prefix and 'error' not in run_results:
            run_results = {'_'.join([prefix, k]): v for k, v in run_results.items()}
        model_results.update(run_results)
    if 'error' not in model_results:
        param_count = model_results.pop('infer_param_count', model_results.pop('train_param_count', 0))
        model_results.setdefault('param_count', param_count)
        model_results.pop('train_param_count', 0)
    return model_results


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []

    if args.model_list:
        args.model = ''
        with open(args.model_list) as f:
            model_names = [line.rstrip() for line in f]
        model_cfgs = [(n, None) for n in model_names]
    elif args.model == 'all':
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
                if not m:
                    continue
                args.model = m
                r = benchmark(args)
                if r:
                    results.append(r)
                time.sleep(10)
        except KeyboardInterrupt as e:
            pass
        sort_key = 'infer_samples_per_sec'
        if 'train' in args.bench:
            sort_key = 'train_samples_per_sec'
        elif 'profile' in args.bench:
            sort_key = 'infer_gmacs'
        results = filter(lambda x: sort_key in x, results)
        results = sorted(results, key=lambda x: x[sort_key], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        results = benchmark(args)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
