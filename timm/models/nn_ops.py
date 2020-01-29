import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

## Assembled CNN Tensorflow Impl
#
# def _bernoulli(shape, mean, seed=None, dtype=tf.float32):
#     return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=dtype, seed=seed)))
#
#
# def dropblock(x, keep_prob, block_size, gamma_scale=1.0, seed=None, name=None,
#               data_format='channels_last', is_training=True):  # pylint: disable=invalid-name
#     """
#     Dropblock layer. For more details, refer to https://arxiv.org/abs/1810.12890
#     :param x: A floating point tensor.
#     :param keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept.
#     :param block_size: The block size to drop
#     :param gamma_scale: The multiplier to gamma.
#     :param seed:  Python integer. Used to create random seeds.
#     :param name: A name for this operation (optional)
#     :param data_format: 'channels_last' or 'channels_first'
#     :param is_training: If False, do nothing.
#     :return: A Tensor of the same shape of x.
#     """
#     if not is_training:
#         return x
#
#     # Early return if nothing needs to be dropped.
#     if (isinstance(keep_prob, float) and keep_prob == 1) or gamma_scale == 0:
#         return x
#
#     with tf.name_scope(name, "dropblock", [x]) as name:
#         if not x.dtype.is_floating:
#             raise ValueError("x has to be a floating point tensor since it's going to"
#                              " be scaled. Got a %s tensor instead." % x.dtype)
#         if isinstance(keep_prob, float) and not 0 < keep_prob <= 1:
#             raise ValueError("keep_prob must be a scalar tensor or a float in the "
#                              "range (0, 1], got %g" % keep_prob)
#
#         br = (block_size - 1) // 2
#         tl = (block_size - 1) - br
#         if data_format == 'channels_last':
#             _, h, w, c = x.shape.as_list()
#             sampling_mask_shape = tf.stack([1, h - block_size + 1, w - block_size + 1, c])
#             pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]
#         elif data_format == 'channels_first':
#             _, c, h, w = x.shape.as_list()
#             sampling_mask_shape = tf.stack([1, c, h - block_size + 1, w - block_size + 1])
#             pad_shape = [[0, 0], [0, 0], [tl, br], [tl, br]]
#         else:
#             raise NotImplementedError
#
#         gamma = (1. - keep_prob) * (w * h) / (block_size ** 2) / ((w - block_size + 1) * (h - block_size + 1))
#         gamma = gamma_scale * gamma
#         mask = _bernoulli(sampling_mask_shape, gamma, seed, tf.float32)
#         mask = tf.pad(mask, pad_shape)
#
#         xdtype_mask = tf.cast(mask, x.dtype)
#         xdtype_mask = tf.layers.max_pooling2d(
#             inputs=xdtype_mask, pool_size=block_size,
#             strides=1, padding='SAME',
#             data_format=data_format)
#
#         xdtype_mask = 1 - xdtype_mask
#         fp32_mask = tf.cast(xdtype_mask, tf.float32)
#         ret = tf.multiply(x, xdtype_mask)
#         float32_mask_size = tf.cast(tf.size(fp32_mask), tf.float32)
#         float32_mask_reduce_sum = tf.reduce_sum(fp32_mask)
#         normalize_factor = tf.cast(float32_mask_size / (float32_mask_reduce_sum + 1e-8), x.dtype)
#         ret = ret * normalize_factor
#         return ret


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
    block_mask = ((2 - seed_drop_rate - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if drop_with_noise:
        normal_noise = torch.randn_like(x)
        x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / (torch.sum(block_mask, dtype=torch.float32) + 1e-7)
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
