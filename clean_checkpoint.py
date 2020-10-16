#!/usr/bin/env python
""" Checkpoint Cleaning Script

Takes training checkpoints with GPU tensors, optimizer state, extra dict keys, etc.
and outputs a CPU  tensor checkpoint with only the `state_dict` along with SHA256
calculation for model zoo compatibility.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import torch
import argparse
import os
import hashlib
import shutil
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch Checkpoint Cleaner')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='output path')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--clean-aux-bn', dest='clean_aux_bn', action='store_true',
                    help='remove auxiliary batch norm layers (from SplitBN training) from checkpoint')

_TEMP_NAME = './_checkpoint.pth'


def main():
    args = parser.parse_args()

    if os.path.exists(args.output):
        print("Error: Output filename ({}) already exists.".format(args.output))
        exit(1)

    # Load an existing checkpoint to CPU, strip everything but the state_dict and re-save
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print("=> Loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        new_state_dict = OrderedDict()
        if isinstance(checkpoint, dict):
            state_dict_key = 'state_dict_ema' if args.use_ema else 'state_dict'
            if state_dict_key in checkpoint:
                state_dict = checkpoint[state_dict_key]
            else:
                state_dict = checkpoint
        else:
            assert False
        for k, v in state_dict.items():
            if args.clean_aux_bn and 'aux_bn' in k:
                # If all aux_bn keys are removed, the SplitBN layers will end up as normal and
                # load with the unmodified model using BatchNorm2d.
                continue
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v
        print("=> Loaded state_dict from '{}'".format(args.checkpoint))

        try:
            torch.save(new_state_dict, _TEMP_NAME, _use_new_zipfile_serialization=False)
        except:
            torch.save(new_state_dict, _TEMP_NAME)

        with open(_TEMP_NAME, 'rb') as f:
            sha_hash = hashlib.sha256(f.read()).hexdigest()

        if args.output:
            checkpoint_root, checkpoint_base = os.path.split(args.output)
            checkpoint_base = os.path.splitext(checkpoint_base)[0]
        else:
            checkpoint_root = ''
            checkpoint_base = os.path.splitext(args.checkpoint)[0]
        final_filename = '-'.join([checkpoint_base, sha_hash[:8]]) + '.pth'
        shutil.move(_TEMP_NAME, os.path.join(checkpoint_root, final_filename))
        print("=> Saved state_dict to '{}, SHA256: {}'".format(final_filename, sha_hash))
    else:
        print("Error: Checkpoint ({}) doesn't exist".format(args.checkpoint))


if __name__ == '__main__':
    main()
