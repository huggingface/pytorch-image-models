#!/usr/bin/env python3
"""Convert released EfficientViM checkpoints (Google Drive `.pth`) to timm format.

The timm port keeps the reference implementation's module names verbatim, so conversion is
mostly a validation pass: the checkpoint is unwrapped if it carries a Lightning
`state_dict` envelope, the tensors of the parallel distillation head are dropped for the
non-distilled variants, and the result is saved as safetensors under the name timm expects
on the Hub.

Human/maintainer step -- weights are only released on Google Drive, so the file must be
downloaded by hand first:
    https://drive.google.com/<EfficientViM release folder>

Usage:
    # non-distilled (drops heads_dist / weights_dist)
    python convert/convert_efficientvim.py \
        --checkpoint EfficientViM_M1_e300.pth \
        --model efficientvim_m1.e300_in1k \
        --output efficientvim_m1.e300_in1k.safetensors

    # distilled (keeps the dist head)
    python convert/convert_efficientvim.py \
        --checkpoint EfficientViM_M1_dist.pth \
        --model efficientvim_m1_dist.in1k \
        --output efficientvim_m1_dist.in1k.safetensors
"""

import argparse

import torch
from safetensors.torch import save_file

import timm
from timm.models.efficientvim import checkpoint_filter_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='path to the downloaded .pth checkpoint')
    parser.add_argument('--model', required=True, help='timm model name, e.g. efficientvim_m1.e300_in1k')
    parser.add_argument('--output', required=True, help='output .safetensors path')
    args = parser.parse_args()

    model = timm.create_model(args.model, pretrained=False)
    # weights_only=False: the released .pth are training checkpoints whose envelope carries
    # non-tensor objects (optimizer / lr_scheduler / config), which weights_only=True rejects.
    # These are the authors' own checkpoints from a trusted source.
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint_filter_fn(state_dict, model)

    result = model.load_state_dict(state_dict, strict=True)
    print(f'loaded {args.model}: {result}')

    save_file(state_dict, args.output, metadata={'format': 'pt'})
    print(f'wrote {args.output} ({len(state_dict)} tensors)')


if __name__ == '__main__':
    main()
