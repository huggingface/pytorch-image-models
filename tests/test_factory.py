import sys
import types

import pytest

from timm.models import (
    generate_default_cfgs, get_arch_pretrained_cfgs, get_deprecated_models, get_pretrained_cfg, is_model_in_modules,
    is_model_pretrained, list_models, list_modules, list_pretrained, model_entrypoint, parse_model_name,
    register_model, register_model_deprecations, safe_model_name,
)


@pytest.mark.parametrize('model_name,expected', [
    # plain timm model names
    ('resnet18', (None, 'resnet18')),
    ('resnet18.a1_in1k', (None, 'resnet18.a1_in1k')),
    # hf-hub, incl. deprecated hf_hub prefix and revision in path
    ('hf-hub:timm/resnet18.a1_in1k', ('hf-hub', 'timm/resnet18.a1_in1k')),
    ('hf_hub:timm/resnet18.a1_in1k', ('hf-hub', 'timm/resnet18.a1_in1k')),
    ('HF-HUB:timm/resnet18.a1_in1k', ('hf-hub', 'timm/resnet18.a1_in1k')),
    ('hf-hub:timm/resnet18.a1_in1k@main', ('hf-hub', 'timm/resnet18.a1_in1k@main')),
    ('hf_hub:user/my_hf_hub_model', ('hf-hub', 'user/my_hf_hub_model')),
    # local-dir, paths must pass through untouched
    ('local-dir:/path/to/model', ('local-dir', '/path/to/model')),
    ('local-dir:./rel/path', ('local-dir', './rel/path')),
    ('local-dir:~/models/my_model', ('local-dir', '~/models/my_model')),
    (r'local-dir:C:\models\my_model', ('local-dir', r'C:\models\my_model')),
    # URL syntax chars are valid in paths, must not be parsed as fragment / query / netloc
    (r'local-dir:C:\##hf-repos\wd-swinv2-tagger-v3', ('local-dir', r'C:\##hf-repos\wd-swinv2-tagger-v3')),
    ('local-dir:/models/model#1', ('local-dir', '/models/model#1')),
    ('local-dir:/models/model?v2', ('local-dir', '/models/model?v2')),
    ('local-dir://server/share/model', ('local-dir', '//server/share/model')),
    (r'local-dir:\\server\share\model', ('local-dir', r'\\server\share\model')),
    # windows extended-length / device / drive-relative forms, colons in the path are not separators
    (r'local-dir:\\?\C:\very\long\path\model', ('local-dir', r'\\?\C:\very\long\path\model')),
    (r'local-dir:\\?\UNC\server\share\model', ('local-dir', r'\\?\UNC\server\share\model')),
    (r'local-dir:\\.\C:\model', ('local-dir', r'\\.\C:\model')),
    (r'local-dir:C:model\rel', ('local-dir', r'C:model\rel')),
    (r'local-dir:C:\dir\model:stream', ('local-dir', r'C:\dir\model:stream')),
])
def test_parse_model_name(model_name, expected):
    assert parse_model_name(model_name) == expected


@pytest.mark.parametrize('model_name', [
    # posix paths
    '/models/resnet18',
    './models/resnet18',
    # windows drive letter, drive-relative, extended-length and device prefixes
    r'C:\models\resnet18',
    r'C:models\resnet18',
    r'C:resnet18',
    r'\\?\C:\very\long\path\resnet18',
    r'\\?\UNC\server\share\resnet18',
    r'\\.\C:\resnet18',
    r'\\?\Volume{GUID}\resnet18',
    r'\\server\share\resnet18',
    # hub repo ids
    'timm/resnet50.a1_in1k',
    'timm/resnet50.a1k',
    'facebook/dinov2-base',
    # ambiguous, a repo id and a one deep relative folder are indistinguishable
    'models/my_model',
])
def test_parse_model_name_no_prefix(model_name):
    # a path / repo id must never silently resolve to a registry model of the same basename,
    # and the error must name both sources as they can't be told apart
    with pytest.raises(ValueError) as exc_info:
        parse_model_name(model_name)
    assert f'hf-hub:{model_name}' in str(exc_info.value)
    assert f'local-dir:{model_name}' in str(exc_info.value)


@pytest.mark.parametrize('model_name', [
    'not-a-source:resnet18',
    'local-dir:',
    'hf-hub:',
])
def test_parse_model_name_invalid(model_name):
    with pytest.raises(ValueError):
        parse_model_name(model_name)


def test_safe_model_name():
    assert safe_model_name('resnet18.a1_in1k') == 'resnet18_a1_in1k'
    assert safe_model_name('hf-hub:timm/resnet18.a1_in1k') == 'timm_resnet18_a1_in1k'
    assert safe_model_name(r'local-dir:C:\##hf-repos\my_model') == 'C____hf_repos_my_model'


@pytest.mark.parametrize('pretrained', [False, True])
def test_list_models_filter_types(pretrained):
    # str, list, tuple, set, and generator filters should all behave the same, incl. w/ tag expansion
    expected = list_models(['resnet50*', 'resnet18*'], pretrained=pretrained, exclude_filters=['*.a1_in1k'])
    assert expected
    include_fns = (
        lambda: ('resnet50*', 'resnet18*'),
        lambda: {'resnet50*', 'resnet18*'},
        lambda: (f for f in ('resnet50*', 'resnet18*')),
    )
    for include_fn in include_fns:
        assert list_models(include_fn(), pretrained=pretrained, exclude_filters='*.a1_in1k') == expected
        assert list_models(include_fn(), pretrained=pretrained, exclude_filters={'*.a1_in1k'}) == expected


def test_list_pretrained_exclude_filters_str():
    as_str = list_pretrained('resnet*', exclude_filters='*.a1_in1k')
    assert as_str
    assert as_str == list_pretrained('resnet*', exclude_filters=['*.a1_in1k'])
    assert not any(m.endswith('.a1_in1k') for m in as_str)


def test_unknown_module_lookup_no_side_effects():
    modules = list_modules()
    assert list_models(module='not_a_timm_module') == []
    assert not is_model_in_modules('resnet50', {'not_a_timm_module'})
    assert list_modules() == modules
    assert list_models(module={'resnet'}) == list_models(module='resnet')
    assert is_model_in_modules('resnet50', 'resnet')


def test_get_arch_pretrained_cfgs_returns_copies():
    assert get_arch_pretrained_cfgs('not_a_timm_model') == {}
    cfgs = get_arch_pretrained_cfgs('resnet50')
    cfg = cfgs['resnet50.a1_in1k']
    orig_mean = cfg.mean
    cfg.mean = (0., 0., 0.)
    assert get_pretrained_cfg('resnet50.a1_in1k').mean == orig_mean


@pytest.fixture
def registry_test_modules():
    # temporary model modules for registration tests, all registry entries are removed afterwards
    from timm.models import _registry
    names = ('registry_test_a', 'registry_test_b')
    mods = {}
    for n in names:
        mods[n] = types.ModuleType(n)
        sys.modules[n] = mods[n]
    registered = []
    yield mods, registered
    for model_name in registered:
        _registry._clear_model_entries(model_name)
        _registry._model_entrypoints.pop(model_name, None)
    for n in names:
        _registry._module_to_models.pop(n, None)
        _registry._module_to_deprecated_models.pop(n, None)
        sys.modules.pop(n, None)


def _register_test_model(mod, name):
    def _fn(pretrained=False, **kwargs):
        return mod.__name__
    _fn.__name__ = name
    _fn.__module__ = mod.__name__
    return register_model(_fn)


def test_register_model_overwrite_clears_stale_entries(registry_test_modules):
    mods, registered = registry_test_modules
    mods['registry_test_a'].default_cfgs = generate_default_cfgs({'registry_test_net.tag_a': {'hf_hub_id': 'timm/'}})
    _register_test_model(mods['registry_test_a'], 'registry_test_net')
    registered.append('registry_test_net')
    assert is_model_pretrained('registry_test_net.tag_a')
    assert list_pretrained('registry_test_net') == ['registry_test_net.tag_a']

    # re-register same name in a different module w/o any pretrained cfgs
    with pytest.warns(UserWarning, match='Overwriting registry_test_net'):
        _register_test_model(mods['registry_test_b'], 'registry_test_net')
    assert list_models(module='registry_test_a') == []
    assert 'registry_test_a' not in list_modules()
    assert list_models(module='registry_test_b') == ['registry_test_net']
    assert not is_model_pretrained('registry_test_net')
    assert not is_model_pretrained('registry_test_net.tag_a')
    assert list_pretrained('registry_test_net') == []
    assert get_pretrained_cfg('registry_test_net') is None
    assert get_arch_pretrained_cfgs('registry_test_net') == {}


def test_register_model_overwrite_replaces_pretrained_cfgs(registry_test_modules):
    mods, registered = registry_test_modules
    mods['registry_test_a'].default_cfgs = generate_default_cfgs({
        'registry_test_net.tag_a': {'hf_hub_id': 'timm/', 'num_classes': 11},
        'registry_test_net.shared': {'hf_hub_id': 'timm/', 'num_classes': 12},
    })
    _register_test_model(mods['registry_test_a'], 'registry_test_net')
    registered.append('registry_test_net')

    # Replace the default tag and keep an existing tag, now without pretrained weights.
    mods['registry_test_b'].default_cfgs = generate_default_cfgs({
        'registry_test_net.tag_b': {'hf_hub_id': 'timm/', 'num_classes': 23},
        'registry_test_net.shared': {'num_classes': 24},
    })
    with pytest.warns(UserWarning, match='Overwriting registry_test_net'):
        _register_test_model(mods['registry_test_b'], 'registry_test_net')

    assert model_entrypoint('registry_test_net')() == 'registry_test_b'
    assert list_models(module='registry_test_a') == []
    assert list_models(module='registry_test_b') == ['registry_test_net']
    assert list_pretrained('registry_test_net') == ['registry_test_net.tag_b']
    assert is_model_pretrained('registry_test_net')
    assert not is_model_pretrained('registry_test_net.tag_a')
    assert not is_model_pretrained('registry_test_net.shared')
    with pytest.raises(RuntimeError, match=r'Invalid pretrained tag \(tag_a\)'):
        get_pretrained_cfg('registry_test_net.tag_a')
    cfgs = get_arch_pretrained_cfgs('registry_test_net')
    assert set(cfgs) == {'registry_test_net.tag_b', 'registry_test_net.shared'}
    assert cfgs['registry_test_net.tag_b'].num_classes == 23
    assert cfgs['registry_test_net.tag_b'].hf_hub_id == 'timm/registry_test_net.tag_b'
    assert cfgs['registry_test_net.shared'].num_classes == 24
    assert get_pretrained_cfg('registry_test_net') == cfgs['registry_test_net.tag_b']


def test_register_model_deprecation_transitions(registry_test_modules):
    mods, registered = registry_test_modules
    mods['registry_test_a'].default_cfgs = generate_default_cfgs({'registry_test_net.tag_a': {'hf_hub_id': 'timm/'}})
    _register_test_model(mods['registry_test_a'], 'registry_test_net')
    registered.append('registry_test_net')
    mods['registry_test_b'].default_cfgs = generate_default_cfgs({
        'registry_test_target.tag_b': {'hf_hub_id': 'timm/', 'num_classes': 23},
    })
    mods['registry_test_b'].registry_test_target = _register_test_model(mods['registry_test_b'], 'registry_test_target')
    registered.append('registry_test_target')

    # Deprecating an existing model must discard its old configs and move its module membership.
    register_model_deprecations('registry_test_b', {'registry_test_net': 'registry_test_target.tag_b'})
    assert list_models('registry_test_net') == []
    assert list_pretrained('registry_test_net') == []
    assert not is_model_pretrained('registry_test_net')
    assert not is_model_pretrained('registry_test_net.tag_a')
    assert get_pretrained_cfg('registry_test_net') is None
    assert get_arch_pretrained_cfgs('registry_test_net') == {}
    assert not is_model_in_modules('registry_test_net', 'registry_test_a')
    assert is_model_in_modules('registry_test_net', 'registry_test_b')
    assert get_deprecated_models()['registry_test_net'] == 'registry_test_target.tag_b'
    assert get_deprecated_models('registry_test_b') == {'registry_test_net': 'registry_test_target.tag_b'}
    with pytest.warns(UserWarning, match='Mapping deprecated model name registry_test_net'):
        assert model_entrypoint('registry_test_net')() == 'registry_test_b'

    # Reusing the deprecated name must restore listings and remove both deprecation mappings.
    mods['registry_test_a'].default_cfgs = generate_default_cfgs({
        'registry_test_net.tag_c': {'hf_hub_id': 'timm/', 'num_classes': 43},
    })
    with pytest.warns(UserWarning, match='Overwriting registry_test_net'):
        _register_test_model(mods['registry_test_a'], 'registry_test_net')
    assert model_entrypoint('registry_test_net')() == 'registry_test_a'
    assert list_models(module='registry_test_a') == ['registry_test_net']
    assert list_models(module='registry_test_b') == ['registry_test_target']
    assert list_pretrained('registry_test_net') == ['registry_test_net.tag_c']
    assert get_pretrained_cfg('registry_test_net').num_classes == 43
    assert 'registry_test_net' not in get_deprecated_models()
    assert get_deprecated_models('registry_test_b') == {}
    assert get_pretrained_cfg('registry_test_target.tag_b').num_classes == 23


def test_list_models_include_tags_no_cfg(registry_test_modules):
    mods, registered = registry_test_modules
    _register_test_model(mods['registry_test_a'], 'registry_test_nocfg')
    registered.append('registry_test_nocfg')
    assert list_models('registry_test_nocfg*') == ['registry_test_nocfg']
    assert list_models('registry_test_nocfg*', include_tags=True) == ['registry_test_nocfg']
    assert list_pretrained('registry_test_nocfg*') == []
