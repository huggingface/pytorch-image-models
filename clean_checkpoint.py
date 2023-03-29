#!/usr/bin/env python3
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
import tempfile
from timm.models import load_state_dict
try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

parser = argparse.ArgumentParser(description='PyTorch Checkpoint Cleaner')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='output path')
parser.add_argument('--no-use-ema', dest='no_use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--no-hash', dest='no_hash', action='store_true',
                    help='no hash in output filename')
parser.add_argument('--clean-aux-bn', dest='clean_aux_bn', action='store_true',
                    help='remove auxiliary batch norm layers (from SplitBN training) from checkpoint')
parser.add_argument('--safetensors', action='store_true',
                    help='Save weights using safetensors instead of the default torch way (pickle).')


def main():
    args = parser.parse_args()

    if os.path.exists(args.output):
        print("Error: Output filename ({}) already exists.".format(args.output))
        exit(1)

    clean_checkpoint(
        args.checkpoint,
        args.output,
        not args.no_use_ema,
        args.no_hash,
        args.clean_aux_bn,
        safe_serialization=args.safetensors,
    )


def clean_checkpoint(
        checkpoint,
        output,
        use_ema=True,
        no_hash=False,
        clean_aux_bn=False,
        safe_serialization: bool=False,
):
    # Load an existing checkpoint to CPU, strip everything but the state_dict and re-save
    if checkpoint and os.path.isfile(checkpoint):
        print("=> Loading checkpoint '{}'".format(checkpoint))
        state_dict = load_state_dict(checkpoint, use_ema=use_ema)
        new_state_dict = {}
        for k, v in state_dict.items():
            if clean_aux_bn and 'aux_bn' in k:
                # If all aux_bn keys are removed, the SplitBN layers will end up as normal and
                # load with the unmodified model using BatchNorm2d.
                continue
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        print("=> Loaded state_dict from '{}'".format(checkpoint))

        ext = ''
        if output:
            checkpoint_root, checkpoint_base = os.path.split(output)
            checkpoint_base, ext = os.path.splitext(checkpoint_base)
        else:
            checkpoint_root = ''
            checkpoint_base = os.path.split(checkpoint)[1]
            checkpoint_base = os.path.splitext(checkpoint_base)[0]

        temp_filename = '__' + checkpoint_base
        if safe_serialization:
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            safetensors.torch.save_file(new_state_dict, temp_filename)
        else:
            torch.save(new_state_dict, temp_filename)

        with open(temp_filename, 'rb') as f:
            sha_hash = hashlib.sha256(f.read()).hexdigest()

        if ext:
            final_ext = ext
        else:
            final_ext = ('.safetensors' if safe_serialization else '.pth')

        if no_hash:
            final_filename = checkpoint_base + final_ext
        else:
            final_filename = '-'.join([checkpoint_base, sha_hash[:8]]) + final_ext

        shutil.move(temp_filename, os.path.join(checkpoint_root, final_filename))
        print("=> Saved state_dict to '{}, SHA256: {}'".format(final_filename, sha_hash))
        return final_filename
    else:
        print("Error: Checkpoint ({}) doesn't exist".format(checkpoint))
        return ''


if __name__ == '__main__':
    main()
