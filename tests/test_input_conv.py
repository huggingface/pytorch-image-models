import pytest
import torch
from torch import nn

from timm.models._builder import load_pretrained
from timm.models._manipulate import adapt_input_conv


def _weights(channels, dtype=torch.float32):
    return torch.arange(2 * channels * 3 * 3, dtype=dtype).reshape(2, channels, 3, 3) / 100


def _reference_channels(weight, source_channels, target_channels):
    if target_channels == source_channels:
        return weight.clone()
    if target_channels == 1:
        return weight.sum(dim=1, keepdim=True)
    return torch.stack([
        weight[:, channel % source_channels] * (source_channels / target_channels)
        for channel in range(target_channels)
    ], dim=1)


@pytest.mark.parametrize('source,target', [(13, 13), (13, 26), (13, 52), (13, 11), (11, 13), (13, 3), (13, 1)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_adapt_non_rgb_input_conv(source, target, dtype):
    weight = _weights(source, dtype=dtype)
    original = weight.clone()

    result = adapt_input_conv(target, weight, base_chans=source)

    expected = _reference_channels(original.float(), source, target).to(dtype)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(weight, original, rtol=0, atol=0)
    assert result.dtype == dtype


@pytest.mark.parametrize('base_channels', [3, 13])
def test_space_to_depth_grayscale_preserves_spatial_groups(base_channels):
    weight = _weights(base_channels * 4)
    expected = torch.stack([
        weight[:, offset:offset + base_channels].sum(dim=1)
        for offset in range(0, base_channels * 4, base_channels)
    ], dim=1)

    result = adapt_input_conv(1, weight, base_chans=base_channels)

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('target', [1, 2, 3, 4, 6])
def test_default_rgb_conversion_is_preserved(target):
    weight = _weights(3)
    original = weight.clone()

    result = adapt_input_conv(target, weight)

    torch.testing.assert_close(result, _reference_channels(original, 3, target))
    torch.testing.assert_close(weight, original, rtol=0, atol=0)


def test_matching_channel_count_preserves_double_precision():
    weight = _weights(13, dtype=torch.float64)

    result = adapt_input_conv(13, weight, base_chans=13)

    torch.testing.assert_close(result, weight, rtol=0, atol=0)


@pytest.mark.parametrize('target,source', [(0, 13), (13, 0), (-1, 13), (13, -1)])
def test_channel_counts_must_be_positive(target, source):
    with pytest.raises(ValueError, match='positive'):
        adapt_input_conv(target, _weights(13), base_chans=source)


class _SensorModel(nn.Module):
    def __init__(self, channels, second_stem=False, linear_stem=False):
        super().__init__()
        self.stem = nn.Linear(channels, 2, bias=False) if linear_stem else nn.Conv2d(channels, 2, 3, bias=False)
        if second_stem:
            self.other_stem = nn.Conv2d(channels, 2, 3, bias=False)
        self.head = nn.Linear(2, 2)


def _load_sensor_model(source_channels, target_channels, second_stem=False, linear_stem=False, input_size=True):
    source = _SensorModel(source_channels, second_stem, linear_stem)
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters()):
            parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) / 100 + index)
    target = _SensorModel(target_channels, second_stem, linear_stem)
    original_target = {key: value.clone() for key, value in target.state_dict().items()}
    state_dict = {key: value.clone() for key, value in source.state_dict().items()}
    cfg = {
        'state_dict': state_dict,
        'first_conv': ('stem', 'other_stem') if second_stem else 'stem',
        'classifier': 'head',
        'num_classes': 2,
    }
    if input_size:
        cfg['input_size'] = (source_channels, 8, 8)
    load_pretrained(target, cfg, num_classes=2, in_chans=target_channels)
    return source, target, original_target, state_dict


@pytest.mark.parametrize('source_channels,target_channels', [
    (13, 13), (13, 26), (13, 52), (13, 11), (11, 13), (13, 3), (13, 1), (3, 1), (3, 6),
])
@pytest.mark.parametrize('second_stem', [False, True])
def test_load_pretrained_uses_source_channel_metadata(source_channels, target_channels, second_stem):
    source, target, _, _ = _load_sensor_model(source_channels, target_channels, second_stem)
    stem_names = ['stem', 'other_stem'] if second_stem else ['stem']
    for name in stem_names:
        source_weight = getattr(source, name).weight
        expected = _reference_channels(source_weight, source_channels, target_channels)
        torch.testing.assert_close(getattr(target, name).weight, expected)
    torch.testing.assert_close(target.head.weight, source.head.weight, rtol=0, atol=0)
    torch.testing.assert_close(target.head.bias, source.head.bias, rtol=0, atol=0)


@pytest.mark.parametrize('frames', [2, 4])
def test_temporal_stacking_preserves_response_to_repeated_frames(frames):
    source, target, _, _ = _load_sensor_model(13, 13 * frames)
    image = torch.linspace(-1, 1, 13 * 8 * 8).reshape(1, 13, 8, 8)

    actual = target.stem(image.repeat(1, frames, 1, 1))
    expected = source.stem(image)

    torch.testing.assert_close(actual, expected)


def test_grayscale_matches_repeating_one_band_for_every_pretrained_channel():
    source, target, _, _ = _load_sensor_model(13, 1)
    image = torch.linspace(-1, 1, 8 * 8).reshape(1, 1, 8, 8)

    torch.testing.assert_close(target.stem(image), source.stem(image.repeat(1, 13, 1, 1)))


def test_missing_source_metadata_retains_rgb_default():
    source, target, _, _ = _load_sensor_model(3, 6, input_size=False)
    torch.testing.assert_close(target.stem.weight, _reference_channels(source.stem.weight, 3, 6))


def test_unsupported_non_rgb_linear_stem_keeps_existing_fallback():
    source, target, original, _ = _load_sensor_model(13, 3, linear_stem=True)
    torch.testing.assert_close(target.stem.weight, original['stem.weight'], rtol=0, atol=0)
    torch.testing.assert_close(target.head.weight, source.head.weight, rtol=0, atol=0)


def test_non_grayscale_space_to_depth_conversion_remains_explicitly_unsupported():
    with pytest.raises(NotImplementedError):
        adapt_input_conv(26, _weights(52), base_chans=13)


@pytest.mark.parametrize('frames', [1, 2])
def test_create_model_transfers_multispectral_checkpoint(frames):
    import timm

    source = timm.create_model('resnet18', pretrained=False, in_chans=13, num_classes=2).eval()
    cfg = dict(
        source.pretrained_cfg,
        state_dict=source.state_dict(),
        input_size=(13, 32, 32),
        num_classes=2,
    )
    target = timm.create_model(
        'resnet18', pretrained=True, in_chans=13 * frames, num_classes=2, pretrained_cfg=cfg,
    ).eval()

    for name, weight in source.state_dict().items():
        expected = _reference_channels(weight, 13, 13 * frames) if name == 'conv1.weight' else weight
        torch.testing.assert_close(target.state_dict()[name], expected, rtol=0, atol=0)
    image = torch.linspace(-1, 1, 13 * 32 * 32).reshape(1, 13, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(target(image.repeat(1, frames, 1, 1)), source(image))
