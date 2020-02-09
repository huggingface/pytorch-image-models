import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def drop_block_2d(x, drop_prob=0.1, block_size=7, gamma_scale=1.0, drop_with_noise=False):
    _, _, height, width = x.shape
    total_size = width * height
    clipped_block_size = min(block_size, min(width, height))
    # seed_drop_rate, the gamma parameter
    seed_drop_rate = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (width - block_size + 1) *
            (height - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(width).to(x.device), torch.arange(height).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < width - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < height - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, height, width)).float()

    uniform_noise = torch.rand_like(x, dtype=torch.float32)
    block_mask = ((2 - seed_drop_rate - valid_block + uniform_noise) >= 1).float()
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if drop_with_noise:
        normal_noise = torch.randn_like(x)
        x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / (torch.sum(block_mask) + 1e-7)
        x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        return drop_block_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise)


def drop_path(x, drop_prob=0.):
    """Drop paths (Stochastic Depth) per sample (when applied in residual blocks)."""
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand((x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.ModuleDict):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        return drop_path(x, self.drop_prob)
