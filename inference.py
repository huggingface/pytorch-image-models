"""Sample PyTorch Inference script
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import torch
import torch.autograd as autograd
import torch.utils.data as data

import model_factory
from dataset import Dataset


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--restore-checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true',
                    help='use multiple-gpus')
parser.add_argument('--no-test-pool', dest='test_time_pool', action='store_false',
                    help='use pre-trained model')


def main():
    args = parser.parse_args()

    # create model
    num_classes = 1000
    model = model_factory.create_model(
        args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        test_time_pool=args.test_time_pool)

    # resume from a checkpoint
    if args.restore_checkpoint and os.path.isfile(args.restore_checkpoint):
        print("=> loading checkpoint '{}'".format(args.restore_checkpoint))
        checkpoint = torch.load(args.restore_checkpoint)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}'".format(args.restore_checkpoint))
    elif not args.pretrained:
        print("=> no checkpoint found at '{}'".format(args.restore_checkpoint))
        exit(1)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    transforms = model_factory.get_transforms_eval(
        args.model,
        args.img_size)

    dataset = Dataset(
        args.data,
        transforms)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model.eval()

    batch_time = AverageMeter()
    end = time.time()
    top5_ids = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            top5 = labels.topk(5)[1]
            top5_ids.append(top5.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('Predict: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    top5_ids = np.concatenate(top5_ids, axis=0).squeeze()

    with open(os.path.join(args.output_dir, './top5_ids.csv'), 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, top5_ids):
            filename = os.path.basename(filename)
            out_file.write('{0},{1},{2},{3},{4},{5}\n'.format(
                filename, label[0], label[1], label[2], label[3], label[4]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
