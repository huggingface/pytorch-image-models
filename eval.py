#!/usr/bin/env python
"""PyTorch Evaluation Script

An example evaluation script that outputs results of model evaluation.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)

--- Usage: ---

model = ClassificationModel('configs/eval.yaml')
img = Image.open("image.jpg")
out = model.eval(img)
print(out)

"""
import yaml
from fire import Fire
from addict import Dict

import torch
from torchvision import transforms

from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

torch.backends.cudnn.benchmark = True


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


class ClassificationModel:
    def __init__(self, config_path: str):
        self.args, self.args_text = _parse_args(config_path)

        # might as well try to do something useful...
        self.args.pretrained = self.args.pretrained or not self.args.checkpoint

        # create model
        self.model = create_model(
            self.args.model,
            num_classes=self.args.num_classes,
            in_chans=3,
            pretrained=self.args.pretrained,
            checkpoint_path=self.args.checkpoint)
        self.softmax = torch.nn.Softmax(dim=1)

        mean = self.args.mean if self.args.mean is not None else IMAGENET_DEFAULT_MEAN
        std = self.args.std if self.args.std is not None else IMAGENET_DEFAULT_STD
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.args.img_size),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std)),
        ])

        if self.args.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.args.num_gpu))).cuda()
        else:
            self.model = self.model.cuda()
            # self.model = self.model.cpu()
        self.model.eval()

    def eval(self, input):
        with torch.no_grad():
            # for OpenCV input
            # input = Image.fromarray(np.uint8(input)).convert('RGB')
            input = self.loader(input).float()
            input = input.cuda()

            labels = self.model(input[None, ...])
            labels = self.softmax(labels)
            labels = labels.cpu()
            return labels.numpy()
