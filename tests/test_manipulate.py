import math

import pytest
import torch

from timm.models._manipulate import adapt_input_conv


@pytest.mark.base
@pytest.mark.parametrize(
    'base_chans, in_chans',
    [
        (3, 5),
        (13, 26),
        (13, 11),
    ],
)
def test_adapt_input_conv_repeats_and_scales_non_rgb_weights(base_chans, in_chans):
    weight = torch.arange(2 * base_chans, dtype=torch.float32).reshape(2, base_chans, 1, 1)
    original = weight.clone()

    adapted = adapt_input_conv(in_chans, weight, base_chans=base_chans)
    repeat = math.ceil(in_chans / base_chans)
    expected = original.repeat(1, repeat, 1, 1)[:, :in_chans] * (base_chans / float(in_chans))

    assert adapted.shape == (2, in_chans, 1, 1)
    torch.testing.assert_close(adapted, expected)


@pytest.mark.base
def test_adapt_input_conv_preserves_space2depth_grouping():
    weight = torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12, 1, 1)

    adapted = adapt_input_conv(1, weight, base_chans=3)
    expected = weight.reshape(2, 4, 3, 1, 1).sum(dim=2)

    assert adapted.shape == (2, 4, 1, 1)
    torch.testing.assert_close(adapted, expected)


@pytest.mark.base
def test_adapt_input_conv_preserves_dtype_and_validates_base_channels():
    weight = torch.ones(2, 13, 1, 1, dtype=torch.float16)

    adapted = adapt_input_conv(1, weight, base_chans=13)

    assert adapted.dtype == weight.dtype
    assert adapted.shape == (2, 1, 1, 1)
    torch.testing.assert_close(adapted, torch.full_like(adapted, 13))

    with pytest.raises(ValueError, match='base_chans must be positive'):
        adapt_input_conv(1, weight, base_chans=0)


@pytest.mark.base
def test_adapt_input_conv_rejects_unknown_weight_channel_layout():
    weight = torch.randn(2, 5, 1, 1)

    with pytest.raises(NotImplementedError, match='Weight format not supported'):
        adapt_input_conv(7, weight, base_chans=3)
