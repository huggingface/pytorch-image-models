import argparse
import time
from collections import OrderedDict
from datetime import datetime

from dataset import Dataset
from models import model_factory, transforms_imagenet_eval, transforms_imagenet_train
from utils import *
from optim import nadam
import scheduler

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: 224)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-s', '--initial-batch-size', type=int, default=0, metavar='N',
                    help='initial input batch size for training (default: 0)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=int, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--drop', type=float, default=0.0, metavar='DROP',
                    help='Dropout rate (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                    help='weight decay (default: 0.0001)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='path to init checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')


def main():
    args = parser.parse_args()

    if args.output:
        output_base = args.output
    else:
        output_base = './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        str(args.img_size)])
    output_dir = get_outdir(output_base, 'train', exp_name)

    batch_size = args.batch_size
    num_epochs = args.epochs
    torch.manual_seed(args.seed)

    dataset_train = Dataset(
        os.path.join(args.data, 'train'),
        transform=transforms_imagenet_train(args.model))

    loader_train = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    dataset_eval = Dataset(
        os.path.join(args.data, 'validation'),
        transform=transforms_imagenet_eval(args.model))

    loader_eval = data.DataLoader(
        dataset_eval,
        batch_size=4 * args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    model = model_factory.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=1000,
        drop_rate=args.drop,
        global_pool=args.gp,
        checkpoint_path=args.initial_checkpoint)

    # optionally resume from a checkpoint
    start_epoch = 0 if args.start_epoch is None else args.start_epoch
    optimizer_state = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('module'):
                        name = k[7:]  # remove `module.`
                    else:
                        name = k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                if 'optimizer' in checkpoint:
                    optimizer_state = checkpoint['optimizer']
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                start_epoch = checkpoint['epoch'] if args.start_epoch is None else args.start_epoch
            else:
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return False

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    train_loss_fn = validate_loss_fn = torch.nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = validate_loss_fn.cuda()

    if args.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    elif args.opt.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'nadam':
        optimizer = nadam.Nadam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert False and "Invalid optimizer"
        exit(1)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    updates_per_epoch = len(loader_train)
    if args.sched == 'cosine':
        lr_scheduler = scheduler.CosineLRScheduler(
            optimizer,
            t_initial=100 * updates_per_epoch,
            t_mul=1.0,
            lr_min=0,
            decay_rate=0.5,
            warmup_lr_init=1e-4,
            warmup_updates=1 * updates_per_epoch
        )
    elif args.sched == 'tanh':
        lr_scheduler = scheduler.TanhLRScheduler(
            optimizer,
            t_initial=80 * updates_per_epoch,
            t_mul=1.0,
            lr_min=1e-5,
            decay_rate=0.5,
            warmup_lr_init=.001,
            warmup_t=5 * updates_per_epoch,
            cycle_limit=1
        )
    else:
        lr_scheduler = scheduler.StepLRScheduler(
            optimizer,
            decay_epochs=args.decay_epochs,
            decay_rate=args.decay_rate,
        )

    saver = CheckpointSaver(checkpoint_dir=output_dir)
    best_loss = None
    try:
        for epoch in range(start_epoch, num_epochs):

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir)

            eval_metrics = validate(
                model, loader_eval, validate_loss_fn, args)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch, eval_metrics['eval_loss'])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_loss is None)

            # save proper checkpoint with eval metric
            best_loss = saver.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                epoch=epoch + 1,
                metric=eval_metrics['eval_loss'])

    except KeyboardInterrupt:
        pass
    print('*** Best loss: {0} (epoch {1})'.format(best_loss[1], best_loss[0]))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=''):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input = input.cuda()
        if isinstance(target, (tuple, list)):
            target = [t.cuda() for t in target]
        else:
            target = target.cuda()

        output = model(input)

        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            print('Train: {} [{}/{} ({:.0f}%)]  '
                  'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                  'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                  '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                  'LR: {lr:.4f}  '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                batch_idx, len(loader),
                100. * batch_idx / last_idx,
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input.size(0) / batch_time_m.val,
                rate_avg=input.size(0) / batch_time_m.avg,
                lr=lr,
                data_time=data_time_m))

            if args.save_images:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        if saver is not None and last_batch or batch_idx % args.recovery_interval == 0:
            save_epoch = epoch + 1 if last_batch else epoch
            saver.save_recovery({
                'epoch': save_epoch,
                'arch': args.model,
                'state_dict':  model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                epoch=save_epoch,
                batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    return OrderedDict([('train_loss', losses_m.avg)])


def validate(model, loader, loss_fn, args):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            input = input.cuda()
            if isinstance(target, list):
                target = target[0].cuda()
            else:
                target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            # calc loss
            loss = loss_fn(output, target)
            losses_m.update(loss.item(), input.size(0))

            # metrics
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if last_batch or batch_idx % args.log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
                      'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                    batch_idx, last_idx,
                    batch_time=batch_time_m, loss=losses_m,
                    top1=prec1_m, top5=prec5_m))

    metrics = OrderedDict([('eval_loss', losses_m.avg), ('eval_prec1', prec1_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
