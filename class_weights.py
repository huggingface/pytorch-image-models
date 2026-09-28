#!/usr/bin/env python3
""" Class Weights

Count per-class labels over a dataset split and write the class stats (per-class positive counts and the
sample count) used by train.py `--class-stats`. From the stats train.py derives `--loss-pos-weight neg_pos`,
`--loss-class-weight inv_freq / effective_num` weights (bce, multi-label asl), and the `--loss db` counts.
`--method` previews the weights a method would produce.

Dataset arguments match train.py. Readers that can enumerate targets without images (hfds, image folder
and tar) skip image data, streaming readers (hfids, wds, tfds) iterate the whole split.

Example:
  python class_weights.py --dataset hfds/timm/plant-pathology-2021 --split train \\
    --task multilabel --target-key labels --num-classes 6 --output stats.json
  python train.py ... --task multilabel --class-stats stats.json --loss-pos-weight neg_pos --loss-weight-power 0.5

Hacked together by / Copyright 2026 Ross Wightman
"""
import argparse
import json
import logging
from itertools import islice

import torch

from timm.data import MultiLabelTarget, create_dataset
from timm.loss import CLASS_WEIGHT_METHOD_KINDS, CLASS_WEIGHT_METHODS, compute_class_weights
from timm.utils import setup_default_logging

_logger = logging.getLogger('class_weights')

parser = argparse.ArgumentParser(description='Count per-class labels (class stats) for train.py loss weighting')
parser.add_argument('--data-dir', metavar='DIR', default=None,
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='train',
                    help='dataset split to count (default: train)')
parser.add_argument('--task', default='classification', choices=('classification', 'multilabel'),
                    help='Target type, class indices or multi-label (default: classification)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Number of classes, required for multilabel, else inferred from the dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--input-key', default=None, type=str,
                    help='Dataset key for input images.')
parser.add_argument('--target-key', default=None, type=str,
                    help='Dataset key for target labels, comma separated for one binary field per class.')
parser.add_argument('--target-format', default=None, type=str, choices=('indices', 'multihot'),
                    help='Multi-label target format (default: indices)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--dataset-trust-remote-code', action='store_true', default=False,
                    help='Allow huggingface dataset import to execute code downloaded from the dataset\'s repo.')
parser.add_argument('--max-samples', type=int, default=None, metavar='N',
                    help='Only count the first N samples (default: all)')
parser.add_argument('--output', '-o', default='', type=str, metavar='FILENAME',
                    help='Write the class stats JSON to this file (default: print to stdout)')
parser.add_argument('--method', default=None, choices=CLASS_WEIGHT_METHODS,
                    help='Preview the weights of a method, "neg_pos" for --loss-pos-weight, "inv_freq" and '
                         '"effective_num" for --loss-class-weight (default: None)')
parser.add_argument('--power', type=float, default=1.,
                    help='Preview exponent for neg_pos / inv_freq, as --loss-weight-power (default: 1.)')
parser.add_argument('--beta', type=float, default=0.999,
                    help='Preview beta for effective_num, as --loss-weight-beta (default: 0.999)')
parser.add_argument('--max-weight', type=float, default=None,
                    help='Preview weight cap, as --loss-weight-max (default: None)')


def iter_targets(dataset):
    """Iterate over dataset targets, avoiding image reads and decoding where the reader allows it."""
    reader = getattr(dataset, 'reader', None)
    if reader is not None and hasattr(reader, 'iter_targets'):
        yield from reader.iter_targets()
        return
    if reader is not None and isinstance(getattr(reader, 'targets', None), (list, tuple)):
        yield from reader.targets  # image in tar
        return
    if reader is not None and isinstance(getattr(reader, 'samples', None), (list, tuple)):
        for sample in reader.samples:  # image folder / tar, (path or tarinfo, target)
            yield sample[1]
        return
    if reader is None and getattr(dataset, 'targets', None) is not None:
        yield from dataset.targets  # torchvision
        return

    source = reader if reader is not None else dataset
    if isinstance(dataset, torch.utils.data.IterableDataset) or not hasattr(source, '__getitem__'):
        items = iter(source)
    else:
        items = (source[i] for i in range(len(source)))
    for item in items:
        if hasattr(item[0], 'close'):
            item[0].close()
        yield item[1]


def main():
    setup_default_logging()
    args = parser.parse_args()
    multi_label = args.task == 'multilabel'
    if multi_label and (args.num_classes is None or args.num_classes <= 0):
        parser.error('--task multilabel requires a positive --num-classes.')
    if not multi_label and args.target_format == 'multihot':
        parser.error('--target-format multihot requires --task multilabel.')

    dataset = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.split,
        class_map=args.class_map,
        download=args.dataset_download,
        input_key=args.input_key,
        target_key=args.target_key,
        target_format=args.target_format,
        trust_remote_code=args.dataset_trust_remote_code,
    )
    reader = getattr(dataset, 'reader', None)
    class_to_idx = getattr(reader, 'class_to_idx', None) or getattr(dataset, 'class_to_idx', None) or {}
    num_classes = args.num_classes or len(class_to_idx)
    if not num_classes:
        parser.error('Could not infer the number of classes from the dataset, pass --num-classes.')

    encode = MultiLabelTarget(num_classes, dense=args.target_format == 'multihot') if multi_label else None
    counts = torch.zeros(num_classes, dtype=torch.float64)
    num_samples = 0
    for target in islice(iter_targets(dataset), args.max_samples):
        if multi_label:
            counts += encode(target).double()
        else:
            target = int(target)
            if not 0 <= target < num_classes:
                raise ValueError(f'Target {target} is out of range for {num_classes} classes.')
            counts[target] += 1
        num_samples += 1
    if not num_samples:
        raise RuntimeError('No samples found in the dataset split.')

    names = [None] * num_classes
    for name, index in class_to_idx.items():
        if isinstance(index, int) and 0 <= index < num_classes:
            names[index] = name
    zero = [names[i] if names[i] is not None else i for i in range(num_classes) if counts[i] == 0]
    _logger.info(
        f'Counted {num_samples} samples, {num_classes} classes, '
        f'positives per class min {int(counts.min())} / max {int(counts.max())}.')
    if zero:
        _logger.warning(f'{len(zero)} classes have no positives and are weighted as if they had one: {zero[:20]}')

    if args.method is not None:
        _, weights = compute_class_weights(
            counts,
            num_samples,
            method=args.method,
            power=args.power,
            beta=args.beta,
            max_weight=args.max_weight,
        )
        flag = '--loss-pos-weight' if CLASS_WEIGHT_METHOD_KINDS[args.method] == 'pos_weight' else '--loss-class-weight'
        _logger.info(
            f'{flag} {args.method} preview: min {weights.min():.4g}, max {weights.max():.4g}, '
            f'mean {weights.mean():.4g}: ' + ','.join(f'{float(w):.4g}' for w in weights))

    is_integral = bool((counts == counts.round()).all())
    stats = dict(
        dataset=args.dataset,
        split=args.split,
        task=args.task,
        target_key=args.target_key,
        num_samples=num_samples,
        num_classes=num_classes,
        pos_counts=[int(c) if is_integral else float(c) for c in counts],
        class_names=names if any(n is not None for n in names) else None,
    )
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        _logger.info(f'Wrote {args.output}, pass it to train.py with --class-stats {args.output}')
    else:
        print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
