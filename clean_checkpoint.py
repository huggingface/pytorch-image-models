import torch
import argparse
import os
import hashlib
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--output', default='./cleaned.pth', type=str, metavar='PATH',
                    help='output path')


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
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v
        print("=> Loaded state_dict from '{}'".format(args.checkpoint))

        torch.save(new_state_dict, args.output)
        with open(args.output, 'rb') as f:
            sha_hash = hashlib.sha256(f.read()).hexdigest()
        print("=> Saved state_dict to '{}, SHA256: {}'".format(args.output, sha_hash))
    else:
        print("Error: Checkpoint ({}) doesn't exist".format(args.checkpoint))


if __name__ == '__main__':
    main()
