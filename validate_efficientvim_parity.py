#!/usr/bin/env python3
"""Validate the timm EfficientViM port against a released checkpoint (GPU / human step).

Two checks, both of which must pass before the upstream PR is opened:

1. Structural -- the checkpoint loads through `checkpoint_filter_fn` with no missing and no
   unexpected keys.
2. Numerical -- ImageNet top-1 matches the reference numbers below (bit-exact parity is the
   expectation: the timm port mirrors the reference math and there is no custom kernel to
   diverge from).

Reference top-1 (ImageNet-1k val):

    model                        params   e300   e450   dist
    efficientvim_m1                6.7M  72.9   73.5   74.6
    efficientvim_m2               13.9M  75.4   75.8   76.7
    efficientvim_m3               16.6M  77.6   77.9   79.1
    efficientvim_m4               19.6M  79.4   79.6   80.7   (256x256)

Usage:
    python validate_efficientvim_parity.py --model efficientvim_m1.e300_in1k --data /path/to/imagenet
"""

import argparse

import torch
from timm.data import create_transform, resolve_data_config
from timm.utils import AverageMeter, accuracy

import timm
from timm.models.efficientvim import checkpoint_filter_fn

EXPECTED_TOP1 = {
    'efficientvim_m1.e300_in1k': 72.9,
    'efficientvim_m1.e450_in1k': 73.5,
    'efficientvim_m1_dist.in1k': 74.6,
    'efficientvim_m2.e300_in1k': 75.4,
    'efficientvim_m2.e450_in1k': 75.8,
    'efficientvim_m2_dist.in1k': 76.7,
    'efficientvim_m3.e300_in1k': 77.6,
    'efficientvim_m3.e450_in1k': 77.9,
    'efficientvim_m3_dist.in1k': 79.1,
    'efficientvim_m4.e300_in1k': 79.4,
    'efficientvim_m4.e450_in1k': 79.6,
    'efficientvim_m4_dist.in1k': 80.7,
}


def check_state_dict(model: torch.nn.Module, path: str) -> None:
    """Verify the checkpoint maps onto the model with no missing / unexpected keys."""
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    state_dict = checkpoint_filter_fn(state_dict, model)
    result = model.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys, f'missing keys: {result.missing_keys}'
    assert not result.unexpected_keys, f'unexpected keys: {result.unexpected_keys}'
    print(f'[ok] {path} loads with no missing / unexpected keys')


@torch.inference_mode()
def validate(model: torch.nn.Module, data_path: str, batch_size: int, device: str) -> float:
    from torchvision.datasets import ImageFolder

    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config)
    dataset = ImageFolder(data_path, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = model.to(device).eval()
    top1 = AverageMeter()
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        if isinstance(output, tuple):
            output = output[0]  # distilled training mode is not used for eval
        acc1 = accuracy(output, target, topk=(1,))[0]
        top1.update(acc1.item(), images.size(0))
        if top1.count % (batch_size * 50) == 0:
            print(f'  {top1.count}/{len(dataset)} images, top-1 {top1.avg:.3f}')
    return top1.avg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, choices=sorted(EXPECTED_TOP1), help='timm model name')
    parser.add_argument('--checkpoint', default=None, help='optional local .pth to check structurally first')
    parser.add_argument('--data', default=None, help='ImageNet val directory (labelled subdirs)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = timm.create_model(args.model, pretrained=args.checkpoint is None)
    if args.checkpoint:
        check_state_dict(model, args.checkpoint)

    expected = EXPECTED_TOP1[args.model]
    if not args.data:
        print(f'[skip] no --data given; structural check only. expected top-1: {expected}')
        return

    top1 = validate(model, args.data, args.batch_size, args.device)
    print(f'top-1: {top1:.3f} (reference {expected})')
    if abs(top1 - expected) > 0.1:
        raise SystemExit('PARITY FAIL: top-1 differs from reference by more than 0.1%')
    print('PARITY OK')


if __name__ == '__main__':
    main()
