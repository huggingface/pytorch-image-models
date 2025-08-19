import importlib
import os
import unittest

import torch

from timm.layers import drop

torch_backend = os.environ.get('TORCH_BACKEND')
if torch_backend is not None:
    importlib.import_module(torch_backend)
torch_device = os.environ.get('TORCH_DEVICE', 'cpu')

class Conv2dKernelMidpointMask2d(unittest.TestCase):
    def test_conv2d_kernel_midpoint_mask_odd_bool(self):
        mask = drop.conv2d_kernel_midpoint_mask(shape=(5, 7), kernel=(3, 3), device=torch_device)
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

    def test_conv2d_kernel_midpoint_mask_odd_float_inplace(self):
        mask = torch.tensor(
            [
                [2.0, 1.0, 1.0, 1.0, 1.0, 7.0, 1.0],
                [1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 8.0],
                [9.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 6.0, 1.0, 1.0],
            ],
            device=torch_device,
        )
        drop.conv2d_kernel_midpoint_mask(
            kernel=(3, 3),
            inplace_mask=mask,
        )
        print(mask)
        assert mask.device == torch.device(torch_device)
        assert mask.tolist() == \
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 4.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 5.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]

    def test_conv2d_kernel_midpoint_mask_odd_float(self):
        mask = drop.conv2d_kernel_midpoint_mask(
            shape=(5, 7),
            kernel=(3, 3),
            device=torch_device,
            dtype=torch.float32,
        )
        print(mask)
        assert mask.device == torch.device(torch_device)
        assert mask.tolist() == \
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]

    def test_conv2d_kernel_midpoint_mask_odd_int(self):
        mask = drop.conv2d_kernel_midpoint_mask(
            shape=(5, 7),
            kernel=(3, 3),
            device=torch_device,
            dtype=torch.int32,
        )
        print(mask)
        assert mask.device == torch.device(torch_device)
        assert mask.tolist() == \
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]

    def test_conv2d_kernel_midpoint_mask_even(self):
        mask = drop.conv2d_kernel_midpoint_mask(shape=(5, 7), kernel=(2, 2), device=torch_device)
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
            drop.conv2d_kernel_midpoint_mask(shape=(4, 7), kernel=(5, 5), device=torch_device)
            raise RuntimeError("Expected throw")

        except AssertionError as e:
            assert "kernel=(5, 5) ! <= shape=(4, 7)" in e.args[0]

