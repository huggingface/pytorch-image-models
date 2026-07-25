import pytest

from timm.models import parse_model_name, safe_model_name


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
