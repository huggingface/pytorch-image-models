#!/usr/bin/env python3
"""Extract Gemma4 Vision Encoder weights from a full Gemma4 multimodal model.

Uses safetensors lazy loading to avoid loading the full model into memory.
Converts HF Transformers weight keys to timm format.

Usage:
    python convert/convert_gemma4_vit.py \
        --source google/gemma-4-4b-it \
        --output gemma4_vit_e4b.safetensors

    python convert/convert_gemma4_vit.py \
        --source google/gemma-4-27b-it \
        --output gemma4_vit_31b.safetensors
"""

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file

# Key mapping from HF Transformers to timm format
HF_PREFIXES = ('model.vision_tower.', 'model.vision_model.', 'vision_model.', 'vision_tower.')


def remap_key(key: str) -> str | None:
    """Remap a single HF Transformers key to timm format.

    Returns None if the key should be skipped.
    Preserves ClippableLinear structure (.linear.weight and clamp buffers).
    """
    # Find and strip the HF prefix
    k = None
    for prefix in HF_PREFIXES:
        if key.startswith(prefix):
            k = key[len(prefix) :]
            break
    if k is None:
        return None

    # Skip rotary embedding buffers (recomputed in model)
    if 'rotary_emb' in k:
        return None

    # Remap component names
    k = k.replace('patch_embedder.', 'patch_embed.')
    k = k.replace('encoder.layers.', 'blocks.')
    k = k.replace('.input_layernorm.', '.norm1.')
    k = k.replace('.post_attention_layernorm.', '.norm2.')
    k = k.replace('.pre_feedforward_layernorm.', '.norm3.')
    k = k.replace('.post_feedforward_layernorm.', '.norm4.')
    k = k.replace('.self_attn.', '.attn.')

    # Note: .linear.weight and clamp buffers (input_min etc.) are preserved
    # since timm model uses Gemma4ClippableLinear with the same structure.

    return k


def extract_vision_weights(source: str, output: str, verbose: bool = True):
    """Extract and convert vision encoder weights from a Gemma4 model."""
    if verbose:
        print(f"Source: {source}")
        print(f"Output: {output}")

    # Find all safetensors files in the repo
    if os.path.isdir(source):
        # Local directory
        shard_files = sorted(Path(source).glob("*.safetensors"))
        shard_paths = [str(f) for f in shard_files]
    else:
        # HuggingFace Hub
        all_files = list_repo_files(source)
        shard_files = sorted([f for f in all_files if f.endswith('.safetensors')])
        if verbose:
            print(f"Found {len(shard_files)} safetensors shards")

        shard_paths = []
        for sf in shard_files:
            if verbose:
                print(f"  Downloading {sf}...")
            path = hf_hub_download(source, sf)
            shard_paths.append(path)

    # Extract vision encoder weights
    out_tensors = {}
    total_params = 0

    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Check if this is a vision key (vision_tower or vision_model)
                if 'vision_tower' not in key and 'vision_model' not in key:
                    continue

                new_key = remap_key(key)
                if new_key is None:
                    if verbose:
                        print(f"  SKIP: {key}")
                    continue

                tensor = f.get_tensor(key)
                out_tensors[new_key] = tensor
                total_params += tensor.numel()

                if verbose:
                    print(f"  {key}")
                    print(f"    -> {new_key}  {list(tensor.shape)}  {tensor.dtype}")

    if verbose:
        print(f"\nExtracted {len(out_tensors)} tensors, {total_params:,} parameters")

    # Save
    save_file(out_tensors, output)
    file_size = os.path.getsize(output) / (1024 * 1024)
    if verbose:
        print(f"Saved to {output} ({file_size:.1f} MB)")

    return out_tensors


def create_config(out_tensors: dict, output_dir: str, variant: str):
    """Create a config.json for the extracted model."""
    # Infer config from tensor shapes
    embed_dim = out_tensors['patch_embed.input_proj.weight'].shape[0]
    num_layers = max(int(k.split('.')[1]) for k in out_tensors if k.startswith('blocks.')) + 1

    head_dim = out_tensors['blocks.0.attn.q_norm.weight'].shape[0]
    # Support both ClippableLinear (.linear.weight) and plain Linear (.weight)
    q_key = (
        'blocks.0.attn.q_proj.linear.weight'
        if 'blocks.0.attn.q_proj.linear.weight' in out_tensors
        else 'blocks.0.attn.q_proj.weight'
    )
    q_proj_out = out_tensors[q_key].shape[0]
    num_heads = q_proj_out // head_dim

    gate_key = (
        'blocks.0.mlp.gate_proj.linear.weight'
        if 'blocks.0.mlp.gate_proj.linear.weight' in out_tensors
        else 'blocks.0.mlp.gate_proj.weight'
    )
    intermediate = out_tensors[gate_key].shape[0]
    standardize = 'std_bias' in out_tensors

    config = {
        "architecture": "gemma4_vit",
        "variant": variant,
        "embed_dim": embed_dim,
        "depth": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "patch_size": 16,
        "norm_eps": 1e-6,
        "rope_theta": 100.0,
        "position_embedding_size": 10240,
        "pooling_kernel_size": 3,
        "standardize": standardize,
        "license": "apache-2.0",
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract Gemma4 Vision Encoder weights")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g., google/gemma-4-4b-it)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output safetensors file path",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="auto",
        help="Model variant name for config (auto-detected if not specified)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory to save config.json (default: same as output)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    out_tensors = extract_vision_weights(args.source, args.output, verbose=not args.quiet)

    config_dir = args.config_dir or os.path.dirname(args.output) or "."
    variant = args.variant
    if variant == "auto":
        # Infer from source name
        if "4b" in args.source.lower():
            variant = "e4b"
        elif "27b" in args.source.lower() or "31b" in args.source.lower():
            variant = "31b"
        else:
            variant = "unknown"

    create_config(out_tensors, config_dir, variant)


if __name__ == "__main__":
    main()
