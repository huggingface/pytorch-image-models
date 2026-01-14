#!/usr/bin/env python3
"""k-NN evaluation for self-supervised learning.

Evaluates learned representations by k-nearest neighbors classification.
No training required - just extracts features and classifies by voting.

Usage:
    python validate_knn.py \
        --model vit_tiny_patch16_224 \
        --checkpoint /path/to/checkpoint.pth \
        --data-dir /path/to/imagenet

    # With specific k values
    python validate_knn.py \
        --model vit_tiny_patch16_224 \
        --checkpoint /path/to/checkpoint.pth \
        --data-dir /path/to/imagenet \
        --k 1 5 10 20 100
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.utils import setup_default_logging, set_jit_fuser, ParseKwargs

_logger = logging.getLogger('validate_knn')


parser = argparse.ArgumentParser(description='k-NN evaluation for SSL models')

# Model
parser.add_argument('--model', '-m', metavar='NAME', default='vit_base_patch16_224',
                    help='model architecture (default: vit_base_patch16_224)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none, use pretrained)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='use pretrained weights if no checkpoint specified')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

# Data
parser.add_argument('--data-dir', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='', type=str, metavar='DATASET',
                    help='dataset type (default: ImageFolder)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='allow download of dataset for torch/ datasets')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split to evaluate on (default: validation)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset split for feature bank (default: train)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='number of classes (default: auto from dataset)')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class map file')

# k-NN parameters
parser.add_argument('--k', nargs='+', type=int, default=[1, 5, 10, 20],
                    help='k values for k-NN (default: 1 5 10 20)')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='temperature for weighted k-NN (default: 0.07)')
parser.add_argument('--use-weighted', action='store_true', default=False,
                    help='use distance-weighted voting instead of majority vote')

# Feature extraction
parser.add_argument('--feature-type', default='cls', type=str,
                    choices=['cls', 'avg', 'max'],
                    help='feature extraction type: cls (CLS token), avg (global average), max (global max)')

# DataLoader
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='batch size for feature extraction (default: 256)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--prefetcher', action='store_true', default=False,
                    help='use fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='pin CPU memory in DataLoader')

# Device
parser.add_argument('--device', default='cuda', type=str,
                    help='device to use (default: cuda)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use AMP for feature extraction')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='use channels_last memory format')

# Misc
parser.add_argument('--log-interval', default=50, type=int, metavar='N',
                    help='log every N batches (default: 50)')


def extract_features(
        model,
        loader,
        device,
        feature_type='cls',
        amp=False,
        channels_last=False,
        log_interval=50,
        desc='Extracting',
):
    """Extract features from a dataset.

    Args:
        model: Feature extractor model
        loader: DataLoader for the dataset
        device: Device to use
        feature_type: 'cls' for CLS token, 'avg' for global average, 'max' for global max
        amp: Use automatic mixed precision
        channels_last: Use channels_last memory format
        log_interval: Log progress every N batches
        desc: Description for logging

    Returns:
        Tuple of (features, labels) tensors
    """
    model.eval()

    all_features = []
    all_labels = []

    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16) if amp else torch.no_grad

    num_batches = len(loader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                # Get features - try forward_features first, fall back to forward
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(images)
                else:
                    features = model(images)

                # Handle different output formats
                if isinstance(features, (tuple, list)):
                    features = features[0]

                # Extract features based on type
                if features.dim() == 3:
                    # Sequence output [B, N, D] - typical for ViTs
                    if feature_type == 'cls':
                        features = features[:, 0]  # CLS token
                    elif feature_type == 'avg':
                        features = features.mean(dim=1)  # Global average
                    elif feature_type == 'max':
                        features = features.max(dim=1)[0]  # Global max
                elif features.dim() == 4:
                    # Conv output [B, C, H, W] - typical for CNNs
                    if feature_type == 'avg':
                        features = features.mean(dim=(2, 3))
                    elif feature_type == 'max':
                        features = features.amax(dim=(2, 3))
                    else:
                        features = features.mean(dim=(2, 3))  # Default to avg for CNNs
                # else: assume [B, D] already

                # L2 normalize features
                features = F.normalize(features.float(), dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            if (batch_idx + 1) % log_interval == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                _logger.info(
                    f'{desc}: [{batch_idx + 1}/{num_batches}]  '
                    f'Time: {elapsed:.1f}s  '
                    f'Rate: {(batch_idx + 1) * loader.batch_size / elapsed:.1f} img/s'
                )

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def knn_classify(
        train_features,
        train_labels,
        test_features,
        test_labels,
        k_values,
        num_classes,
        temperature=0.07,
        use_weighted=False,
):
    """Perform k-NN classification.

    Args:
        train_features: Feature bank [N_train, D]
        train_labels: Labels for feature bank [N_train]
        test_features: Query features [N_test, D]
        test_labels: Ground truth labels [N_test]
        k_values: List of k values to evaluate
        num_classes: Number of classes
        temperature: Temperature for weighted voting
        use_weighted: Use distance-weighted voting

    Returns:
        Dictionary mapping k to accuracy
    """
    max_k = max(k_values)
    num_test = test_features.shape[0]

    # Move to GPU for faster computation if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    results = {k: 0 for k in k_values}

    # Process in chunks to avoid OOM
    chunk_size = 256

    for start_idx in range(0, num_test, chunk_size):
        end_idx = min(start_idx + chunk_size, num_test)
        chunk_features = test_features[start_idx:end_idx].to(device)
        chunk_labels = test_labels[start_idx:end_idx].to(device)

        # Compute cosine similarity (features are already L2 normalized)
        similarity = chunk_features @ train_features.T  # [chunk, N_train]

        # Get top-k neighbors
        topk_sim, topk_idx = similarity.topk(max_k, dim=1)  # [chunk, max_k]
        topk_labels = train_labels[topk_idx]  # [chunk, max_k]

        for k in k_values:
            k_labels = topk_labels[:, :k]  # [chunk, k]

            if use_weighted:
                # Distance-weighted voting
                k_sim = topk_sim[:, :k]  # [chunk, k]
                weights = (k_sim / temperature).softmax(dim=1)  # [chunk, k]

                # Weighted vote
                votes = torch.zeros(end_idx - start_idx, num_classes, device=device)
                for i in range(k):
                    votes.scatter_add_(1, k_labels[:, i:i+1], weights[:, i:i+1])
                predictions = votes.argmax(dim=1)
            else:
                # Majority voting
                predictions = k_labels.mode(dim=1)[0]

            correct = (predictions == chunk_labels).sum().item()
            results[k] += correct

    # Convert to accuracy
    for k in k_values:
        results[k] = 100.0 * results[k] / num_test

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()

    device = torch.device(args.device)

    # Create model
    _logger.info(f'Creating model: {args.model}')
    model = timm.create_model(
        args.model,
        pretrained=args.pretrained and not args.checkpoint,
        num_classes=0,  # Remove classifier head
        **args.model_kwargs,
    )

    # Load checkpoint
    if args.checkpoint:
        _logger.info(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove classifier weights if present
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith('head.') and not k.startswith('fc.')}

        msg = model.load_state_dict(state_dict, strict=False)
        _logger.info(f'Loaded checkpoint: {msg}')

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Resolve data config
    data_config = resolve_data_config(vars(args), model=model)
    _logger.info(f'Data config: {data_config}')

    # Create datasets
    _logger.info(f'Creating train dataset from {args.train_split} split')
    train_dataset = create_dataset(
        root=args.data_dir,
        name=args.dataset,
        split=args.train_split,
        download=args.dataset_download,
        class_map=args.class_map,
    )

    _logger.info(f'Creating eval dataset from {args.split} split')
    eval_dataset = create_dataset(
        root=args.data_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        class_map=args.class_map,
    )

    num_classes = args.num_classes or len(train_dataset.classes)
    _logger.info(f'Number of classes: {num_classes}')
    _logger.info(f'Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}')

    # Create data loaders
    train_loader = create_loader(
        train_dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        device=device,
        persistent_workers=args.workers > 0,
    )

    eval_loader = create_loader(
        eval_dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        device=device,
        persistent_workers=args.workers > 0,
    )

    # Extract features
    _logger.info('Extracting train features...')
    train_features, train_labels = extract_features(
        model, train_loader, device,
        feature_type=args.feature_type,
        amp=args.amp,
        channels_last=args.channels_last,
        log_interval=args.log_interval,
        desc='Train',
    )
    _logger.info(f'Train features shape: {train_features.shape}')

    _logger.info('Extracting eval features...')
    eval_features, eval_labels = extract_features(
        model, eval_loader, device,
        feature_type=args.feature_type,
        amp=args.amp,
        channels_last=args.channels_last,
        log_interval=args.log_interval,
        desc='Eval',
    )
    _logger.info(f'Eval features shape: {eval_features.shape}')

    # k-NN classification
    _logger.info(f'Running k-NN classification with k={args.k}...')
    results = knn_classify(
        train_features, train_labels,
        eval_features, eval_labels,
        k_values=args.k,
        num_classes=num_classes,
        temperature=args.temperature,
        use_weighted=args.use_weighted,
    )

    # Report results
    _logger.info('=' * 50)
    _logger.info('k-NN Results:')
    for k, acc in sorted(results.items()):
        _logger.info(f'  k={k:3d}: {acc:.2f}%')
    _logger.info('=' * 50)

    return results


if __name__ == '__main__':
    main()
