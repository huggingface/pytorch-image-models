"""PatchMix 

Papers:
Inter-Instance Similarity Modeling for Contrastive Learning (https://arxiv.org/abs/2306.12243)

Code Reference:
PatchMix: https://github.com/visresearch/patchmix
"""
import torch
import random
import numpy as np


def one_hot(x, num_classes, on_value=1.0, off_value=0.0):
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)


def random_indexes(size):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(x, indexes):
    return torch.gather(x, 1, indexes.unsqueeze(-1).repeat(1, 1, x.shape[-1]))


class PatchMix:
    """PatchMix that applies different params to whole batch

    Args:
        num_classes (int): the number of categories in contrastive learning, it is generally the sum of batch sizes across all nodes.
        prob (float): The probability of performing patch mix. Default is ``0.5``.
        mix_num (int): The number of original images included in each set of mix images. Default is ``2``.
        patch_size (int): size of image patch. Default is ``16``.
        smoothing (float): coefficient of label smoothing Default is ``0.``.
    """

    def __init__(self, num_classes, prob=0.5, mix_num=2, patch_size=16, smoothing=0.0):
        super().__init__()
        self.prob = prob
        self.mix_num = mix_num
        self.patch_size = patch_size
        self.smoothing = smoothing
        self.num_classes = num_classes

    def _shuffle(self, x):
        b, l = x.shape[:2]
        indexes = random_indexes(l)
        forward_indexes = torch.as_tensor(indexes[0], dtype=torch.long).to(x.device)
        forward_indexes = forward_indexes.repeat(b, 1)
        backward_indexes = torch.as_tensor(indexes[1], dtype=torch.long).to(x.device)
        backward_indexes = backward_indexes.repeat(b, 1)
        x = take_indexes(x, forward_indexes)
        return x, forward_indexes, backward_indexes

    def _mix(self, x, m):
        b, l, c = x.shape
        s = l // m
        d = b * m
        l_ = int(s * m)
        # get the image sequence that needs to be mixed, and drop the last.
        mix_x = x[:, :l_]
        mix_x = mix_x.reshape(d, s, c)
        # generate the mix index for mixing patch group.
        ids = torch.arange(d, device=x.device)
        mix_indexes = (ids + ids % m * m) % d
        mix_x = torch.gather(mix_x, 0, mix_indexes.repeat(s, c, 1).permute(-1, 0, 1))
        mix_x = mix_x.reshape(b, l_, c)
        x[:, :l_] = mix_x
        # generate the mix index for mixing target.
        ids = torch.arange(b, device=x.device).view(-1, 1)
        m2o_indexes = (ids + torch.arange(m, device=x.device)) % b
        m2m_indexes = ((ids - m + 1) + torch.arange(m * 2 - 1, device=x.device) + b) % b
        return x, m2o_indexes, m2m_indexes

    def __call__(self, x, target):
        """
            img (Tensor): Image to be mixed.
            target (Tensor): target for contrastive learning.

        Returns:
            x (Tensor): mixed image.
            m2o_target (Tensor): target between mixed images and original images in infoNCE loss.
            m2m_target (Tensor): target between mixed images and mixed images in infoNCE loss.
        """
        b, c, h, w = x.shape
        m = self.mix_num
        # We only use patch mix when m is greater than 1
        use_mix = random.random() < self.prob and m > 1
        if use_mix:
            p = self.patch_size
            n_h = h // p
            n_w = w // p
            # b c (w p1) (h p2) -> b (w h) (c p1 p2)
            x = x.reshape(b, c, n_h, p, n_w, p).permute(0, 2, 4, 1, 3, 5).reshape(b, n_h * n_w, c * p * p)
            x, _, backward_indexes = self._shuffle(x)
            x, m2o_indexes, m2m_indexes = self._mix(x, m)
            x = take_indexes(x, backward_indexes)
            # b (w h) (c p1 p2) -> b c (w p1) (h p2)
            x = x.reshape(b, n_h, n_w, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, n_h * p, n_w * p)
        else:
            m = 1
            m2o_indexes = target.view(-1, 1)
            m2m_indexes = m2o_indexes

        # get mixed target for mix-to-org loss and mix-to-mix loss
        m2o_target = target[m2o_indexes]
        m2m_target = target[m2m_indexes]

        off_value = self.smoothing / self.num_classes
        true_num = m2o_target.shape[1]
        on_value = (1.0 - self.smoothing) / true_num + off_value
        m2o_target = one_hot(m2o_target, self.num_classes, on_value, off_value)

        ids = torch.arange(m2m_target.shape[1], device=x.device)
        weights = 1.0 - torch.abs(m - ids - 1) / m
        on_value = (1.0 - self.smoothing) * weights / m + off_value
        m2m_target = one_hot(m2m_target, self.num_classes, on_value.expand([m2m_target.shape[0], -1]), off_value)

        return x, m2o_target, m2m_target
