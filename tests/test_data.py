import numpy as np
import pytest
import torch

from timm.data import create_transform
from timm.data.mixup import rand_bbox_minmax


@pytest.mark.parametrize('count', [None, 4])
def test_rand_bbox_minmax_can_reach_bottom_right_border(monkeypatch, count):
    def sample_last(low, high, size=None):
        result = np.asarray(high) - 1
        return result if size is None else np.broadcast_to(result, size)

    monkeypatch.setattr(np.random, 'randint', sample_last)

    yl, yu, xl, xu = rand_bbox_minmax((3, 5, 7), (0.5, 0.75), count=count)

    assert np.all(yl == 3)
    assert np.all(yu == 5)
    assert np.all(xl == 3)
    assert np.all(xu == 7)


def test_naflex_eval_patchify_can_preserve_spatial_patch_dimensions():
    common_kwargs = dict(
        input_size=(3, 16, 16),
        is_training=False,
        naflex=True,
        patch_size=(2, 2),
        max_seq_len=16,
        patchify=True,
    )
    image = torch.rand(3, 8, 8)

    spatial = create_transform(**common_kwargs, patchify_flatten=False)(image)
    flattened = create_transform(**common_kwargs)(image)

    assert spatial['patches'].ndim == 4
    assert spatial['patches'].shape[1:] == (2, 2, 3)
    assert flattened['patches'].ndim == 2
    assert flattened['patches'].shape[-1] == 2 * 2 * 3
