import pytest
import torch

from timm.models._hub import load_state_dict_from_path

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False


def _write_ckpt(path, value):
    state_dict = {'weight': torch.full((2,), float(value))}
    if path.suffix == '.safetensors':
        if not _has_safetensors:
            pytest.skip('safetensors not installed')
        safetensors.torch.save_file(state_dict, path)
    else:
        torch.save(state_dict, path)


def _value(state_dict):
    return int(state_dict['weight'][0].item())


def test_load_state_dict_from_path_prefers_preferred_file(tmp_path):
    # a file from _PREFERRED_FILES wins over any other checkpoint in the folder
    _write_ckpt(tmp_path / 'model.safetensors.notused.bin', 1)
    _write_ckpt(tmp_path / 'model.safetensors', 2)
    _write_ckpt(tmp_path / 'pytorch_model.bin', 3)
    assert _value(load_state_dict_from_path(tmp_path, weights_only=True)) == 2


def test_load_state_dict_from_path_ext_priority(tmp_path):
    # no preferred file name, so extension priority decides, .safetensors over .pth over .bin
    _write_ckpt(tmp_path / 'weights.bin', 1)
    _write_ckpt(tmp_path / 'weights.pth', 2)
    assert _value(load_state_dict_from_path(tmp_path, weights_only=True)) == 2

    _write_ckpt(tmp_path / 'weights.safetensors', 3)
    assert _value(load_state_dict_from_path(tmp_path, weights_only=True)) == 3


def test_load_state_dict_from_path_multiple_same_ext_warns(tmp_path, caplog):
    # multiple checkpoints of one extension must warn (not raise NameError) and use the first sorted
    _write_ckpt(tmp_path / 'a_weights.pth', 1)
    _write_ckpt(tmp_path / 'b_weights.pth', 2)
    with caplog.at_level('WARNING'):
        state_dict = load_state_dict_from_path(tmp_path, weights_only=True)
    assert _value(state_dict) == 1
    assert 'a_weights.pth' in caplog.text and 'b_weights.pth' in caplog.text


def test_load_state_dict_from_path_accepts_str(tmp_path):
    _write_ckpt(tmp_path / 'model.safetensors', 1)
    assert _value(load_state_dict_from_path(str(tmp_path), weights_only=True)) == 1


def test_load_state_dict_from_path_no_checkpoint(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    with pytest.raises(RuntimeError, match='No suitable checkpoints'):
        load_state_dict_from_path(tmp_path)
