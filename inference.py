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

from models import create_model, apply_test_time_pool
from data import Dataset, create_loader, get_mean_and_std
from utils import AverageMeter

torch.backends.cudnn.benchmark = True

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
parser.add_argument('--no-test-pool', dest='test_time_pool', action='store_false',
                    help='use pre-trained model')


def main():
    args = parser.parse_args()

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    print('Model %s created, param count: %d' %
          (args.model, sum([m.numel() for m in model.parameters()])))

    data_mean, data_std = get_mean_and_std(model, args)
    model, test_time_pool = apply_test_time_pool(model, args)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    loader = create_loader(
        Dataset(args.data),
        img_size=args.img_size,
        batch_size=args.batch_size,
        use_prefetcher=True,
        mean=data_mean,
        std=data_std,
        num_workers=args.workers)

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
        filenames = loader.dataset.filenames()
        for filename, label in zip(filenames, top5_ids):
            filename = os.path.basename(filename)
            out_file.write('{0},{1},{2},{3},{4},{5}\n'.format(
                filename, label[0], label[1], label[2], label[3], label[4]))


if __name__ == '__main__':
    main()
