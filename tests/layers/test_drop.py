import importlib
import os
import unittest

import torch

from timm.layers import drop

torch_backend = os.environ.get('TORCH_BACKEND')
if torch_backend is not None:
    importlib.import_module(torch_backend)
torch_device = os.environ.get('TORCH_DEVICE', 'cpu')

class ClipMaskTests(unittest.TestCase):
    def test_clip_mask_2d_odd(self):
        mask = drop.clip_mask_2d(h=5, w=7, kernel=3, device=torch_device)
        print(mask)
        assert mask.device == torch.device(torch_device)
        assert mask.tolist() == \
            [
                [False, False, False, False, False, False, False],
                [False, True, True, True, True, True, False],
                [False, True, True, True, True, True, False],
                [False, True, True, True, True, True, False],
                [False, False, False, False, False, False, False],
            ]

    def test_clip_mask_2d_even(self):
        mask = drop.clip_mask_2d(h=5, w=7, kernel=2, device=torch_device)
        print(mask)
        assert mask.device == torch.device(torch_device)
        # TODO: This is a suprising result; should even kernels be forbidden?
        assert mask.tolist() == \
            [
                [False, False, False, False, False, False, False],
                [False, True, True, True, True, True, True],
                [False, True, True, True, True, True, True],
                [False, True, True, True, True, True, True],
                [False, True, True, True, True, True, True],
            ]

    def test_clip_mask_2d_kernel_too_big(self):
        try:
            drop.clip_mask_2d(h=4, w=7, kernel=5, device=torch_device)
            raise RuntimeError("Expected throw")

        except AssertionError as e:
            assert "kernel=5 > min(h=4, w=7)" in e.args[0]

