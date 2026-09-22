import copy
import json
from contextlib import nullcontext

import pytest
import torch

import timm
from timm.data import resolve_data_config, resolve_input_data_config
from timm.layers import DropBlock2d, DropPath
from timm.models import get_model_args, get_model_input_config, get_model_traits, resolve_model_input_args
from timm.models._builder import load_pretrained
from timm.models._hub import save_for_hf


_HAS_DEVICE_CONTEXT = hasattr(torch.device, '__enter__')
_CFG_DEVICE = 'meta' if _HAS_DEVICE_CONTEXT else 'cpu'


def _vit(**kwargs):
    return timm.create_model(
        'vit_tiny_patch16_224',
        **dict(img_size=(32, 48), embed_dim=32, depth=1, num_heads=4, **kwargs),
    )


def test_script_resolution_preserves_source_config_and_legacy_resolver():
    args = dict(in_chans=4, img_size=32, mean=(0.25,), std=(0.5,))
    model = timm.create_model(
        **resolve_model_input_args(
            'vit_tiny_patch16_224',
            args,
            embed_dim=32,
            depth=1,
            num_heads=4,
        )
    )
    cfg = resolve_input_data_config(model, args)
    assert model.patch_embed.img_size == (32, 32)
    assert model.in_chans == 4
    assert cfg['input_size'] == (4, 32, 32)
    assert cfg['mean'] == (0.25,) * 4
    assert cfg['std'] == (0.5,) * 4
    assert model.pretrained_cfg['input_size'] == (3, 224, 224)
    assert resolve_data_config(model=model)['input_size'] == (3, 224, 224)


@pytest.mark.parametrize(
    'args,kwargs',
    [
        (dict(in_chans=4, input_size=(3, 32, 32)), {}),
        (dict(in_chans=4), dict(in_chans=3)),
        (dict(img_size=32, input_size=(3, 48, 48)), {}),
        (dict(img_size=32), dict(img_size=48)),
    ],
)
def test_script_construction_rejects_conflicting_inputs(args, kwargs):
    with pytest.raises(ValueError, match='Conflicting input'):
        resolve_model_input_args('vit_tiny_patch16_224', args, **kwargs)


def test_script_resolution_defaults_to_multispectral_source():
    kwargs = resolve_model_input_args(
        'resnet18',
        pretrained_cfg_overlay=dict(input_size=(13, 32, 32), mean=(0.5,), std=(0.5,)),
    )
    with torch.device(_CFG_DEVICE) if _HAS_DEVICE_CONTEXT else nullcontext():
        model = timm.create_model(**kwargs, device=_CFG_DEVICE)
    assert model.in_chans == 13
    assert resolve_input_data_config(model)['input_size'] == (13, 32, 32)


def test_flexible_model_uses_test_size_and_explicit_data_overrides():
    model = _vit(dynamic_img_size=True, pretrained_cfg_overlay=dict(test_input_size=(3, 256, 256)))
    assert not get_model_input_config(model)['fixed_input_size']
    assert resolve_input_data_config(model, use_test_size=True)['input_size'] == (3, 256, 256)
    args = dict(img_size=64, model_kwargs=dict(img_size=(32, 48), dynamic_img_size=True))
    assert resolve_input_data_config(model, args)['input_size'] == (3, 64, 64)
    created = timm.create_model(
        **resolve_model_input_args(
            'vit_tiny_patch16_224',
            args,
            **args['model_kwargs'],
            embed_dim=32,
            depth=1,
            num_heads=4,
        )
    )
    assert created.patch_embed.img_size == (32, 48)


@pytest.mark.parametrize('args', [dict(in_chans=4), dict(img_size=64), dict(mean=(0.5, 0.5))])
def test_runtime_resolution_rejects_incompatible_explicit_requests(args):
    with pytest.raises(ValueError):
        resolve_input_data_config(_vit(), args)


def test_conv_model_image_size_is_preprocessing_only():
    kwargs = resolve_model_input_args('resnet18', dict(input_size=(4, 96, 64)))
    assert 'img_size' not in kwargs
    with torch.device(_CFG_DEVICE) if _HAS_DEVICE_CONTEXT else nullcontext():
        model = timm.create_model(**kwargs, device=_CFG_DEVICE)
    assert resolve_input_data_config(model, dict(input_size=(4, 96, 64)))['input_size'] == (4, 96, 64)


def test_matching_stem_shape_loads_despite_stale_channel_metadata(caplog):
    source = torch.nn.Sequential(torch.nn.Conv2d(4, 2, 3))
    target = torch.nn.Sequential(torch.nn.Conv2d(4, 2, 3))
    cfg = dict(input_size=(3, 32, 32), first_conv='0', state_dict=source.state_dict())
    load_pretrained(target, cfg, in_chans=4)
    torch.testing.assert_close(target[0].weight, source[0].weight, rtol=0, atol=0)
    assert 'already match' in caplog.text


def test_channel_disagreement_is_not_inferred_from_weight_ratios():
    source = torch.nn.Sequential(torch.nn.Conv2d(4, 2, 3))
    target = torch.nn.Sequential(torch.nn.Conv2d(8, 2, 3))
    cfg = dict(input_size=(3, 32, 32), first_conv='0', state_dict=source.state_dict())
    with pytest.raises(RuntimeError, match='Correct the source pretrained_cfg'):
        load_pretrained(target, cfg, in_chans=8)


@pytest.mark.parametrize('legacy_metadata', [False, True])
def test_adapted_resized_model_hub_roundtrip(tmp_path, legacy_metadata):
    source = _vit(in_chans=4, num_classes=11).eval()
    cfg_before = copy.deepcopy(source.pretrained_cfg)
    source.set_input_size(img_size=(48, 64), patch_size=8)
    source.reset_classifier(7)
    assert get_model_input_config(source)['input_size'] == (4, 48, 64)
    assert resolve_input_data_config(source)['input_size'] == (4, 48, 64)
    data_config = dict(mean=(0.1, 0.2, 0.3, 0.4), std=(0.5,) * 4, crop_pct=1.0)
    save_for_hf(source, tmp_path, data_config=data_config)
    cfg = json.loads((tmp_path / 'config.json').read_text())
    assert cfg['pretrained_cfg']['input_size'] == [4, 48, 64]
    assert cfg['pretrained_cfg']['mean'] == list(data_config['mean'])
    assert cfg['pretrained_cfg']['num_classes'] == cfg['num_classes'] == 7
    assert source.pretrained_cfg == cfg_before
    if legacy_metadata:
        cfg['pretrained_cfg']['input_size'] = [3, 224, 224]
        (tmp_path / 'config.json').write_text(json.dumps(cfg))

    restored = timm.create_model('local-dir:' + str(tmp_path), pretrained=True).eval()
    assert restored.in_chans == 4
    assert restored.pretrained_cfg['input_size'] == (4, 48, 64)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    x = torch.randn(1, 4, 48, 64)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), source(x), rtol=0, atol=0)

    adapted = timm.create_model('local-dir:' + str(tmp_path), pretrained=True, in_chans=8)
    torch.testing.assert_close(
        adapted.patch_embed.proj.weight,
        source.patch_embed.proj.weight.repeat(1, 2, 1, 1) / 2,
    )
    assert adapted.pretrained_cfg['input_size'][0] == 4
    script_args = resolve_model_input_args('local-dir:' + str(tmp_path), dict(in_chans=8, img_size=80))
    script_model = timm.create_model(**script_args, pretrained=True)
    assert script_model.in_chans == 8
    assert script_model.patch_embed.img_size == (80, 80)
    assert script_model.pretrained_cfg['input_size'] == (4, 48, 64)


def test_export_requires_serializable_architecture_options(tmp_path):
    model = _vit(norm_layer=torch.nn.LayerNorm)
    with pytest.raises(ValueError, match="'norm_layer' is not serializable"):
        save_for_hf(model, tmp_path)
    save_for_hf(model, tmp_path, model_args=dict(norm_layer='layernorm'))
    restored = timm.create_model('local-dir:' + str(tmp_path), pretrained=True)
    assert isinstance(restored.norm, torch.nn.LayerNorm)


@pytest.mark.parametrize('nested', [False, True])
def test_export_rejects_conflicting_model_metadata(tmp_path, nested):
    model = _vit(in_chans=4)
    kwargs = dict(model_args=dict(in_chans=3))
    if nested:
        kwargs = dict(model_config=kwargs)
    with pytest.raises(ValueError, match='disagrees with the current model'):
        save_for_hf(model, tmp_path, **kwargs)


@pytest.mark.parametrize('use_naflex', [False, True])
def test_flexible_model_export_keeps_nominal_grid_and_preprocessing_separate(tmp_path, use_naflex):
    kwargs = dict(use_naflex=True) if use_naflex else dict(dynamic_img_size=True)
    model = _vit(num_classes=7, **kwargs)
    first_conv = 'embeds.proj' if use_naflex else 'patch_embed.proj'
    assert get_model_traits(model)['first_conv'] == first_conv
    save_for_hf(model, tmp_path, data_config=dict(input_size=(3, 64, 64)))
    restored = timm.create_model('local-dir:' + str(tmp_path), pretrained=True)
    assert not get_model_input_config(restored)['fixed_input_size']
    assert get_model_traits(restored)['first_conv'] == restored.pretrained_cfg['first_conv'] == first_conv
    assert restored.pretrained_cfg['input_size'] == (3, 64, 64)
    assert tuple(get_model_args(restored)['img_size']) == (32, 48)
    assert resolve_input_data_config(restored)['input_size'] == (3, 64, 64)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('name', ['swin_tiny_patch4_window7_224', 'swinv2_tiny_window8_256', 'swinv2_cr_tiny_ns_224'])
def test_windowed_model_resize_roundtrip(tmp_path, name):
    model = timm.create_model(
        name,
        img_size=64,
        embed_dim=16,
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 8),
        num_classes=7,
    ).eval()
    model.set_input_size(img_size=96, window_size=4)
    model.set_input_size(window_size=3)
    save_for_hf(model, tmp_path)
    restored = timm.create_model('local-dir:' + str(tmp_path), pretrained=True).eval()
    assert resolve_input_data_config(restored)['input_size'] == (3, 96, 96)
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_feature_wrapper_retains_input_contract():
    with torch.device(_CFG_DEVICE) if _HAS_DEVICE_CONTEXT else nullcontext():
        model = timm.create_model('resnet18', in_chans=4, features_only=True, device=_CFG_DEVICE)
    assert model.in_chans == 4
    traits = get_model_traits(model)
    assert traits['in_chans'] == 4
    assert traits['first_conv'] == 'conv1'
    assert 'num_classes' not in traits
    assert get_model_input_config(model)['input_size'][0] == 4


def test_model_traits_are_independent_and_follow_resize_and_head_reset():
    model = _vit(in_chans=4, num_classes=7)
    other = _vit()
    original = get_model_traits(model)
    other_before = get_model_traits(other)
    assert original == dict(
        img_size=(32, 48),
        fixed_input_size=True,
        first_conv='patch_embed.proj',
        in_chans=4,
        num_classes=7,
    )
    snapshot = get_model_traits(model)
    snapshot.update(img_size=(999, 999), num_classes=999)
    assert get_model_traits(model) == original

    model.set_input_size(img_size=(48, 64))
    model.reset_classifier(5)
    assert get_model_traits(model) == dict(original, img_size=(48, 64), num_classes=5)
    assert get_model_traits(other) == other_before
    assert get_model_args(model)['img_size'] == (48, 64)
    assert get_model_args(model)['num_classes'] == 5


def test_export_tracks_reset_pool_and_omits_in_memory_weight_source(tmp_path):
    model = timm.create_model('resnet18', num_classes=7)
    model.reset_classifier(3, global_pool='catavgmax')
    model.pretrained_cfg['state_dict'] = {'unused': torch.ones(1)}
    save_for_hf(model, tmp_path)
    restored = timm.create_model('local-dir:' + str(tmp_path), pretrained=True)
    assert restored.global_pool.pool_type == 'catavgmax'
    assert restored.num_classes == 3
    torch.testing.assert_close(restored.fc.weight, model.fc.weight, rtol=0, atol=0)


@pytest.mark.parametrize('vit', [False, True])
def test_export_does_not_inherit_training_regularization(tmp_path, vit):
    rates = dict(drop_rate=0.2, drop_path_rate=0.1)
    if vit:
        rates.update(pos_drop_rate=0.1, patch_drop_rate=0.1, proj_drop_rate=0.1, attn_drop_rate=0.1)
        model = _vit(num_classes=7, **rates)
    else:
        rates['drop_block_rate'] = 0.1
        model = timm.create_model('resnet18', num_classes=7, **rates)
    assert not rates.keys() & get_model_args(model).keys()
    save_for_hf(model, tmp_path)
    cfg = json.loads((tmp_path / 'config.json').read_text())
    assert not rates.keys() & cfg['model_args'].keys()
    restored = timm.create_model('local-dir:' + str(tmp_path), pretrained=True)
    for module in restored.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.p == 0
        elif isinstance(module, (DropPath, DropBlock2d)):
            assert module.drop_prob == 0
    if vit:
        assert isinstance(restored.patch_drop, torch.nn.Identity)
    else:
        assert restored.drop_rate == 0
