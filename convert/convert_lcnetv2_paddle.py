""" Convert PP-LCNetV2 weights from PaddleClas to timm

Checkpoints: https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/docs/en/models/PP-LCNetV2_en.md
NOTE: `paddlepaddle` is required to unpickle the .pdparams files, it is not in requirements.txt

Usage:
    python convert/convert_lcnetv2_paddle.py PPLCNetV2_base_pretrained.pdparams \
        --model lcnetv2_base --output lcnetv2_base.pth
"""
import argparse
import re
from typing import Any, Dict

import numpy as np
import torch

import timm

parser = argparse.ArgumentParser(description='Convert PaddleClas PP-LCNetV2 weights')
parser.add_argument('checkpoint', metavar='PDPARAMS', help='path of the PaddleClas .pdparams checkpoint')
parser.add_argument('--model', default='lcnetv2_base', help='name of the target timm model')
parser.add_argument('--output', default='./converted.pth', help='output path of the converted checkpoint')
parser.add_argument('--dropout-prob', type=float, default=0.2,
                    help='dropout_prob the Paddle model was created with (see PPLCNetV2_* entrypoints)')


def _remap_key(k: str) -> str:
    k = k.replace('.pw_conv_1.', '.pw_conv.0.')
    k = k.replace('.pw_conv_2.', '.pw_conv.1.')
    k = k.replace('.se.conv1.', '.se.fc1.')
    k = k.replace('.se.conv2.', '.se.fc2.')
    k = k.replace('.bn._mean', '.bn.running_mean')
    k = k.replace('.bn._variance', '.bn.running_var')
    k = re.sub(r'^last_conv\.', 'conv_head.', k)
    k = re.sub(r'^fc\.', 'classifier.', k)
    return k


def convert_state_dict(paddle_state_dict: Dict[str, Any], dropout_prob: float = 0.2) -> Dict[str, torch.Tensor]:
    # Paddle keeps the dropout probability as a float32 attribute, so the constant it multiplies by at
    # inference is float32(1 - p). Rounding it here keeps the fold below exact for float32 checkpoints
    # (where torch would round the scalar anyway) and for float64 ones alike.
    keep_prob = float(np.float32(1. - dropout_prob))

    out_dict = {}
    for k, v in paddle_state_dict.items():
        # the released checkpoints carry the re-parameterized depthwise conv alongside the branches it was
        # folded from, timm keeps the branches and folds on demand via timm.utils.reparameterize_model()
        repped = re.fullmatch(r'(.*)\.dw_conv\.(?:weight|bias)', k)
        if repped and f'{repped.group(1)}.dw_conv_list.0.conv.weight' in paddle_state_dict:
            continue

        v = torch.from_numpy(v)
        if k == 'fc.weight':
            v = v.transpose(0, 1).contiguous()  # paddle Linear weights are [in_features, out_features]
        elif k == 'last_conv.weight':
            # Paddle applies dropout with mode='downscale_in_infer', which scales the pre-logits features by
            # (1 - p) at inference. The 1x1 conv is bias free and followed by ReLU, so folding the scale into
            # its weights is exact and lets timm keep a standard nn.Dropout / F.dropout head.
            v = v * keep_prob

        out_dict[_remap_key(k)] = v
    return out_dict


def main():
    args = parser.parse_args()

    import paddle  # noqa: only needed to read the checkpoint
    paddle_state_dict = {k: v.numpy() for k, v in paddle.load(args.checkpoint).items()}

    state_dict = convert_state_dict(paddle_state_dict, dropout_prob=args.dropout_prob)
    model = timm.create_model(args.model, pretrained=False)
    model.load_state_dict(state_dict)
    torch.save(state_dict, args.output)
    print(f'Converted {len(state_dict)} tensors from {args.checkpoint} to {args.output}')


if __name__ == '__main__':
    main()
