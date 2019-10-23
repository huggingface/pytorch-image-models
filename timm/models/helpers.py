import torch
import torch.utils.model_zoo as model_zoo
import os
import logging
from collections import OrderedDict


def load_checkpoint(model, checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            state_dict_key = 'state_dict'
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        logging.info("Loaded {} from checkpoint '{}'".format(state_dict_key or 'weights', checkpoint_path))
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def resume_checkpoint(model, checkpoint_path):
    other_state = {}
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            if 'optimizer' in checkpoint:
                other_state['optimizer'] = checkpoint['optimizer']
            if 'amp' in checkpoint:
                other_state['amp'] = checkpoint['amp']
            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
            logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return other_state, resume_epoch
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        logging.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        logging.info('Converting first conv (%s) from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        state_dict[conv1_name + '.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        assert False, "Invalid in_chans for pretrained weights"

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)






