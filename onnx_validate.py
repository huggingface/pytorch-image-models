""" ONNX-runtime validation script

This script was created to verify accuracy and performance of exported ONNX
models running with the onnxruntime. It utilizes the PyTorch dataloader/processing
pipeline for a fair comparison against the originals.

Copyright 2020 Ross Wightman
"""
import argparse
import numpy as np
import onnxruntime
from timm.data import create_loader, resolve_data_config, create_dataset
from timm.utils import AverageMeter
import time

parser = argparse.ArgumentParser(description='ONNX Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--onnx-input', default='', type=str, metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-output-opt', default='', type=str, metavar='PATH',
                    help='path to output optimized onnx graph')
parser.add_argument('--profile', action='store_true', default=False,
                    help='Enable profiler output.')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--crop-pct', type=float, default=None, metavar='PCT',
                    help='Override default crop pct of 0.875')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main():
    args = parser.parse_args()
    args.gpu_id = 0

    # Set graph optimization level
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.profile:
        sess_options.enable_profiling = True
    if args.onnx_output_opt:
        sess_options.optimized_model_filepath = args.onnx_output_opt

    session = onnxruntime.InferenceSession(args.onnx_input, sess_options)

    data_config = resolve_data_config(vars(args))
    loader = create_loader(
        create_dataset('', args.data),
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct']
    )

    input_name = session.get_inputs()[0].name

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for i, (input, target) in enumerate(loader):
        # run the net and return prediction
        output = session.run([], {input_name: input.data.numpy()})
        output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy_np(output, target.numpy())
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                f'Test: [{i}/{len(loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {input.size(0) / batch_time.avg:.3f}/s, '
                f'{100 * batch_time.avg / input.size(0):.3f} ms/sample) \t'
                f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
            )

    print(f' * Prec@1 {top1.avg:.3f} ({100-top1.avg:.3f}) Prec@5 {top5.avg:.3f} ({100.-top5.avg:.3f})')


def accuracy_np(output, target):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5], target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


if __name__ == '__main__':
    main()
