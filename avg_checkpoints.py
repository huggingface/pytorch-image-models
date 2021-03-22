#!/usr/bin/env python3
""" Checkpoint Averaging Script

This script averages all model weights for checkpoints in specified path that match
the specified filter wildcard. All checkpoints must be from the exact same model.

For any hope of decent results, the checkpoints should be from the same or child
(via resumes) training session. This can be viewed as similar to maintaining running
EMA (exponential moving average) of the model weights or performing SWA (stochastic
weight averaging), but post-training.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import torch
import argparse
import os
import glob
import hashlib
from timm.models.helpers import load_state_dict

parser = argparse.ArgumentParser(description='PyTorch Checkpoint Averager')
parser.add_argument('--input', default='', type=str, metavar='PATH',
                    help='path to base input folder containing checkpoints')
parser.add_argument('--filter', default='*.pth.tar', type=str, metavar='WILDCARD',
                    help='checkpoint filter (path wildcard)')
parser.add_argument('--output', default='./averaged.pth', type=str, metavar='PATH',
                    help='output filename')
parser.add_argument('--no-use-ema', dest='no_use_ema', action='store_true',
                    help='Force not using ema version of weights (if present)')
parser.add_argument('--no-sort', dest='no_sort', action='store_true',
                    help='Do not sort and select by checkpoint metric, also makes "n" argument irrelevant')
parser.add_argument('-n', type=int, default=10, metavar='N',
                    help='Number of checkpoints to average')


def checkpoint_metric(checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return {}
    print("=> Extracting metric from checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    metric = None
    if 'metric' in checkpoint:
        metric = checkpoint['metric']
    return metric


def main():
    args = parser.parse_args()
    # by default use the EMA weights (if present)
    args.use_ema = not args.no_use_ema
    # by default sort by checkpoint metric (if present) and avg top n checkpoints
    args.sort = not args.no_sort

    if os.path.exists(args.output):
        print("Error: Output filename ({}) already exists.".format(args.output))
        exit(1)

    pattern = args.input
    if not args.input.endswith(os.path.sep) and not args.filter.startswith(os.path.sep):
        pattern += os.path.sep
    pattern += args.filter
    checkpoints = glob.glob(pattern, recursive=True)

    if args.sort:
        checkpoint_metrics = []
        for c in checkpoints:
            metric = checkpoint_metric(c)
            if metric is not None:
                checkpoint_metrics.append((metric, c))
        checkpoint_metrics = list(sorted(checkpoint_metrics))
        checkpoint_metrics = checkpoint_metrics[-args.n:]
        print("Selected checkpoints:")
        [print(m, c) for m, c in checkpoint_metrics]
        avg_checkpoints = [c for m, c in checkpoint_metrics]
    else:
        avg_checkpoints = checkpoints
        print("Selected checkpoints:")
        [print(c) for c in checkpoints]

    avg_state_dict = {}
    avg_counts = {}
    for c in avg_checkpoints:
        new_state_dict = load_state_dict(c, args.use_ema)
        if not new_state_dict:
            print("Error: Checkpoint ({}) doesn't exist".format(args.checkpoint))
            continue

        for k, v in new_state_dict.items():
            if k not in avg_state_dict:
                avg_state_dict[k] = v.clone().to(dtype=torch.float64)
                avg_counts[k] = 1
            else:
                avg_state_dict[k] += v.to(dtype=torch.float64)
                avg_counts[k] += 1

    for k, v in avg_state_dict.items():
        v.div_(avg_counts[k])

    # float32 overflow seems unlikely based on weights seen to date, but who knows
    float32_info = torch.finfo(torch.float32)
    final_state_dict = {}
    for k, v in avg_state_dict.items():
        v = v.clamp(float32_info.min, float32_info.max)
        final_state_dict[k] = v.to(dtype=torch.float32)

    try:
        torch.save(final_state_dict, args.output, _use_new_zipfile_serialization=False)
    except:
        torch.save(final_state_dict, args.output)

    with open(args.output, 'rb') as f:
        sha_hash = hashlib.sha256(f.read()).hexdigest()
    print("=> Saved state_dict to '{}, SHA256: {}'".format(args.output, sha_hash))


if __name__ == '__main__':
    main()
