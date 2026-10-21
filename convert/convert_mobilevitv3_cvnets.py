"""Convert MobileViTv3 CVNets classification checkpoints to timm.

The official MobileViTv3 repository has separate v1 (XXS/XS/S) and v2
(width-multiplier) implementations. This converter supports both state-dict
layouts and requires an explicit target timm model. The v1 checkpoints were
trained with BGR inputs; their stem weights are permuted so converted models
accept standard RGB inputs. The v2 checkpoints were trained with RGB inputs.

Usage:
    python convert/convert_mobilevitv3_cvnets.py CHECKPOINT \
        --model mobilevitv3_xxs --version v1 --output /tmp/mobilevitv3_xxs.pth
    python convert/convert_mobilevitv3_cvnets.py CHECKPOINT \
        --model mobilevitv3_050 --version v2 --output /tmp/mobilevitv3_050.pth
    python convert/convert_mobilevitv3_cvnets.py CHECKPOINT \
        --model mobilevitv3_050.cvnets_in1k_ft_in1k_384 --version v2 \
        --output /tmp/mobilevitv3_050_ft_384.pth
"""
import argparse
import re
from collections import OrderedDict
from typing import Mapping

import torch

import timm


def _remap_common_key(key: str) -> str:
    if key.startswith('conv_1.block.conv.'):
        return key.replace('conv_1.block.conv.', 'stem.conv.', 1)
    if key.startswith('conv_1.block.norm.'):
        return key.replace('conv_1.block.norm.', 'stem.bn.', 1)

    # Inverted residual layers have the same three operations in both official
    # checkpoints; only the module names differ from timm's ByobNet blocks.
    match = re.fullmatch(
        r'layer_(\d+)\.(\d+)\.block\.(exp_1x1|conv_3x3|red_1x1)\.block\.(conv|norm)\.(.+)',
        key,
    )
    if match:
        stage, block, op, layer, suffix = match.groups()
        op_map = {'exp_1x1': 'conv1_1x1', 'conv_3x3': 'conv2_kxk', 'red_1x1': 'conv3_1x1'}
        layer = 'conv' if layer == 'conv' else 'bn'
        return f'stages.{int(stage) - 1}.{block}.{op_map[op]}.{layer}.{suffix}'
    return key


def _remap_v1_key(key: str) -> str:
    key = _remap_common_key(key)

    match = re.fullmatch(r'layer_(\d+)\.(\d+)\.local_rep\.conv_3x3\.block\.(conv|norm)\.(.+)', key)
    if match:
        stage, block, layer, suffix = match.groups()
        layer = 'conv' if layer == 'conv' else 'bn'
        return f'stages.{int(stage) - 1}.{block}.conv_kxk.{layer}.{suffix}'

    match = re.fullmatch(r'layer_(\d+)\.(\d+)\.local_rep\.conv_1x1\.block\.conv\.(.+)', key)
    if match:
        stage, block, suffix = match.groups()
        return f'stages.{int(stage) - 1}.{block}.conv_1x1.weight' if suffix == 'weight' else key

    match = re.fullmatch(r'layer_(\d+)\.(\d+)\.global_rep\.(\d+)\.(.+)', key)
    if match:
        stage, block, depth, subkey = match.groups()
        prefix = f'stages.{int(stage) - 1}.{block}'
        transformer_prefix = f'{prefix}.transformer.{depth}'

        match = re.fullmatch(r'pre_norm_mha\.0\.(weight|bias)', subkey)
        if match:
            return f'{transformer_prefix}.norm1.{match.group(1)}'
        match = re.fullmatch(r'pre_norm_mha\.1\.(qkv_proj|out_proj)\.(weight|bias)', subkey)
        if match:
            layer = {'qkv_proj': 'qkv', 'out_proj': 'proj'}[match.group(1)]
            return f'{transformer_prefix}.attn.{layer}.{match.group(2)}'
        match = re.fullmatch(r'pre_norm_ffn\.0\.(weight|bias)', subkey)
        if match:
            return f'{transformer_prefix}.norm2.{match.group(1)}'
        match = re.fullmatch(r'pre_norm_ffn\.(1|4)\.(weight|bias)', subkey)
        if match:
            layer = {'1': 'fc1', '4': 'fc2'}[match.group(1)]
            return f'{transformer_prefix}.mlp.{layer}.{match.group(2)}'
        if re.fullmatch(r'(weight|bias)', subkey):
            return f'{prefix}.norm.{subkey}'

    match = re.fullmatch(r'layer_(\d+)\.(\d+)\.(conv_proj|fusion)\.block\.(conv|norm)\.(.+)', key)
    if match:
        stage, block, op, layer, suffix = match.groups()
        op = {'conv_proj': 'conv_proj', 'fusion': 'conv_fusion'}[op]
        layer = 'conv' if layer == 'conv' else 'bn'
        return f'stages.{int(stage) - 1}.{block}.{op}.{layer}.{suffix}'

    if key.startswith('conv_1x1_exp.block.conv.'):
        return key.replace('conv_1x1_exp.block.conv.', 'final_conv.conv.', 1)
    if key.startswith('conv_1x1_exp.block.norm.'):
        return key.replace('conv_1x1_exp.block.norm.', 'final_conv.bn.', 1)
    if key.startswith('classifier.fc.'):
        return key.replace('classifier.fc.', 'head.fc.', 1)
    return key


def _remap_v2_key(key: str) -> str:
    key = _remap_common_key(key)

    match = re.fullmatch(r'layer_(\d+)\.(\d+)\.local_rep\.(0|1)\.block\.(conv|norm)\.(.+)', key)
    if match:
        stage, block, local_op, layer, suffix = match.groups()
        prefix = f'stages.{int(stage) - 1}.{block}'
        if local_op == '0':
            layer = 'conv' if layer == 'conv' else 'bn'
            return f'{prefix}.conv_kxk.{layer}.{suffix}'
        if layer == 'conv' and suffix == 'weight':
            return f'{prefix}.conv_1x1.weight'
        return key

    match = re.fullmatch(r'layer_(\d+)\.(\d+)\.global_rep\.(\d+)\.(.+)', key)
    if match:
        stage, block, depth, subkey = match.groups()
        prefix = f'stages.{int(stage) - 1}.{block}'
        transformer_prefix = f'{prefix}.transformer.{depth}'

        match = re.fullmatch(r'pre_norm_attn\.0\.(weight|bias)', subkey)
        if match:
            return f'{transformer_prefix}.norm1.{match.group(1)}'
        match = re.fullmatch(r'pre_norm_attn\.1\.(qkv_proj|out_proj)\.block\.conv\.(weight|bias)', subkey)
        if match:
            layer = {'qkv_proj': 'qkv_proj', 'out_proj': 'out_proj'}[match.group(1)]
            return f'{transformer_prefix}.attn.{layer}.{match.group(2)}'
        match = re.fullmatch(r'pre_norm_ffn\.0\.(weight|bias)', subkey)
        if match:
            return f'{transformer_prefix}.norm2.{match.group(1)}'
        match = re.fullmatch(r'pre_norm_ffn\.(1|3)\.block\.conv\.(weight|bias)', subkey)
        if match:
            layer = {'1': 'fc1', '3': 'fc2'}[match.group(1)]
            return f'{transformer_prefix}.mlp.{layer}.{match.group(2)}'
        if re.fullmatch(r'(weight|bias)', subkey):
            return f'{prefix}.norm.{subkey}'

    match = re.fullmatch(r'layer_(\d+)\.(\d+)\.conv_proj\.block\.(conv|norm)\.(.+)', key)
    if match:
        stage, block, layer, suffix = match.groups()
        layer = 'conv' if layer == 'conv' else 'bn'
        return f'stages.{int(stage) - 1}.{block}.conv_proj.{layer}.{suffix}'

    if key.startswith('classifier.1.'):
        return key.replace('classifier.1.', 'head.fc.', 1)
    return key


def convert_state_dict(
        official_state_dict: Mapping[str, torch.Tensor], version: str = 'v1'
) -> OrderedDict[str, torch.Tensor]:
    """Remap an official MobileViTv3 classification state dict to timm names."""
    if version not in ('v1', 'v2'):
        raise ValueError(f'Unsupported MobileViTv3 version: {version}')
    remap_key = _remap_v1_key if version == 'v1' else _remap_v2_key

    converted = OrderedDict()
    source_keys = {}
    for key, value in official_state_dict.items():
        new_key = remap_key(key)
        if new_key == key and not key.startswith('stages.'):
            raise KeyError(f'No {version} key mapping for official tensor: {key}')
        if new_key in converted:
            raise KeyError(f'Multiple official tensors map to {new_key}: {source_keys[new_key]}, {key}')
        converted[new_key] = value
        source_keys[new_key] = key

    if version == 'v1':
        stem_weight = converted['stem.conv.weight']
        if stem_weight.shape[1] != 3:
            raise ValueError(f'Expected a 3-channel v1 stem, got shape {tuple(stem_weight.shape)}')
        # Official v1 uses BGR tensors. Reorder input-channel filters so timm
        # can consume RGB tensors without a runtime channel-reversal transform.
        converted['stem.conv.weight'] = stem_weight[:, [2, 1, 0]].contiguous()

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert official MobileViTv3 CVNets weights to timm')
    parser.add_argument('checkpoint', help='path to official checkpoint_ema_best.pt')
    parser.add_argument('--model', required=True, help='target timm model name')
    parser.add_argument('--version', choices=('v1', 'v2'), required=True, help='official MobileViTv3 lineage')
    parser.add_argument('--output', default='./converted_mobilevitv3.pth', help='converted state-dict output path')
    args = parser.parse_args()

    official_state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_dict = convert_state_dict(official_state_dict, version=args.version)

    model = timm.create_model(args.model, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    torch.save(state_dict, args.output)
    print(f'Loaded {len(state_dict)} tensors strictly for {args.model}')
    print(f'Saved converted checkpoint to {args.output}')


if __name__ == '__main__':
    main()
