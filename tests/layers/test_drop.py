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
    def test_conv2d_kernel_midpoint_mask_odd(self):
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


    def test_conv2d_kernel_midpoint_mask_even(self):
        mask = drop.conv2d_kernel_midpoint_mask(
            shape=(5, 7),
            kernel=(2, 2),
            device=torch_device,
            dtype=torch.float32,
        )
        print(mask)
        assert mask.device == torch.device(torch_device)
        assert mask.tolist() == \
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]

    def test_clip_mask_2d_kernel_too_big(self):
        try:
            drop.conv2d_kernel_midpoint_mask(
                shape=(4, 7),
                kernel=(5, 5),
                device=torch_device,
                dtype=torch.float32,
            )
            raise RuntimeError("Expected throw")

        except AssertionError as e:
            assert "kernel=(5, 5) ! <= shape=(4, 7)" in e.args[0]


class DropBlock2dDropFilterTest(unittest.TestCase):
    def test_drop_filter(self):
        selection = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=torch_device,
        ).unsqueeze(0).unsqueeze(0)

        result = drop.drop_block_2d_drop_filter_(
            selection=selection,
            kernel=(2, 3),
            partial_edge_blocks=False
        ).squeeze()
        print(result)
        assert result.device == torch.device(torch_device)
        assert result.tolist() == \
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]

    def test_drop_filter_partial_edge_blocks(self):
        selection = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=torch_device,
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        result = drop.drop_block_2d_drop_filter_(
            selection=selection,
            kernel=(2, 3),
            partial_edge_blocks=True
        ).squeeze()
        print(result)
        assert result.device == torch.device(torch_device)
        assert result.tolist() == \
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ]

class DropBlock2dTest(unittest.TestCase):
    def test_drop_block_2d(self):
        tensor = torch.ones((1, 1, 200, 300), device=torch_device)

        drop_prob=0.1
        keep_prob = 1.0 - drop_prob

        result = drop.drop_block_2d(
            tensor,
            drop_prob=drop_prob,
            with_noise=True,
        ).squeeze()

        numel = float(result.numel())
        unchanged = float(len(result[result == 1.0]))
        keep_ratio = unchanged / numel

        assert abs(keep_ratio - keep_prob) < 0.05, \
                f"abs({keep_ratio=} - {keep_prob=}) ! < 0.05"

