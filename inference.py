#!/usr/bin/env python
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import logging
import yaml
from fire import Fire
from addict import Dict

import numpy as np
import torch

from timm.models import create_model, apply_test_time_pool
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


def _update_config(config, params):
    for k, v in params.items():
        *path, key = k.split(".")
        config.update({k: v})
        print(f"Overwriting {k} = {v} (was {config.get(key)})")
    return config


def _fit(config_path, **kwargs):
    with open(config_path) as stream:
        base_config = yaml.safe_load(stream)

    if "config" in kwargs.keys():
        cfg_path = kwargs["config"]
        with open(cfg_path) as cfg:
            cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)

        merged_cfg = _update_config(base_config, cfg_yaml)
    else:
        merged_cfg = base_config

    update_cfg = _update_config(merged_cfg, kwargs)
    return update_cfg


def _parse_args(config_path):
    args = Dict(Fire(_fit(config_path)))

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args('configs/inference.yaml')
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    loader = create_loader(
        Dataset(args.data),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            topk = labels.topk(k)[1]
            topk_ids.append(topk.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    topk_ids = np.concatenate(topk_ids, axis=0).squeeze()

    with open(os.path.join(args.output_dir, './topk_ids.csv'), 'w') as out_file:
        filenames = loader.dataset.filenames(basename=True)
        for filename, label in zip(filenames, topk_ids):
            out_file.write('{0},{1},{2},{3},{4},{5}\n'.format(
                filename, label[0], label[1], label[2], label[3], label[4]))


if __name__ == '__main__':
    main()
