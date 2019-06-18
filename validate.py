from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import csv
import glob
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict

from models import create_model, apply_test_time_pool, load_checkpoint
from data import Dataset, create_loader, resolve_data_config
from utils import accuracy, AverageMeter, natural_key

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
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
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--tf-preprocessing', dest='tf_preprocessing', action='store_true',
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')


def validate(args):

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained)

    if args.checkpoint and not args.pretrained:
        load_checkpoint(model, args.checkpoint, args.use_ema)
    else:
        args.pretrained = True  # might as well try to validate something...

    param_count = sum([m.numel() for m in model.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(model, args)
    model, test_time_pool = apply_test_time_pool(model, data_config, args)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    loader = create_loader(
        Dataset(args.data, load_bytes=args.tf_preprocessing),
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else data_config['crop_pct'],
        tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {rate_avg:.3f}/s) \t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    rate_avg=input.size(0) / batch_time.avg,
                    loss=losses, top1=top1, top5=top5))

    results = OrderedDict(
        top1=round(top1.avg, 3), top1_err=round(100 - top1.avg, 3),
        top5=round(top5.avg, 3), top5_err=round(100 - top5.avg, 3),
        param_count=round(param_count / 1e6, 2))

    print(' * Prec@1 {:.3f} ({:.3f}) Prec@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results


def main():
    args = parser.parse_args()
    if args.model == 'all':
        # validate all models in a list of names with pretrained checkpoints
        args.pretrained = True
        # FIXME just an example list, need to add model name collections for
        # batch testing of various pretrained combinations by arg string
        models = ['tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3']
        model_cfgs = [(n, '') for n in models]
    elif os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        model_cfgs = []

    if len(model_cfgs):
        header_written = False
        with open('./results-all.csv', mode='w') as cf:
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                result.update(validate(args))
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                dw = csv.DictWriter(cf, fieldnames=result.keys())
                if not header_written:
                    dw.writeheader()
                    header_written = True
                dw.writerow(result)
                cf.flush()
    else:
        validate(args)


if __name__ == '__main__':
    main()
