import argparse
import inspect

import pytest
import torch

from timm.models._helpers import load_state_dict, resume_checkpoint


_HAS_WEIGHTS_ONLY = 'weights_only' in inspect.signature(torch.load).parameters
_HAS_SAFE_GLOBALS = hasattr(torch.serialization, 'safe_globals')


class _CustomPayload:
    def __init__(self, value: int = 1):
        self.value = value


@pytest.mark.skipif(
    not (_HAS_WEIGHTS_ONLY and _HAS_SAFE_GLOBALS),
    reason='requires torch.load(weights_only=...) with safe_globals support',
)
def test_weights_only_allows_argparse_namespace(tmp_path):
    checkpoint_path = tmp_path / 'namespace_ckpt.pth'
    checkpoint = {
        'state_dict': {'layer.weight': torch.randn(2, 2)},
        'args': argparse.Namespace(model='test-model'),
    }
    torch.save(checkpoint, checkpoint_path)

    state_dict = load_state_dict(checkpoint_path)
    assert 'layer.weight' in state_dict


@pytest.mark.skipif(not _HAS_WEIGHTS_ONLY, reason='requires torch.load(weights_only=...) support')
def test_weights_only_blocks_non_allowlisted_globals(tmp_path):
    checkpoint_path = tmp_path / 'custom_ckpt.pth'
    checkpoint = {
        'state_dict': {'layer.weight': torch.randn(2, 2)},
        'args': _CustomPayload(3),
    }
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match='No automatic unsafe pickle fallback is performed'):
        load_state_dict(checkpoint_path)


@pytest.mark.skipif(
    not (_HAS_WEIGHTS_ONLY and _HAS_SAFE_GLOBALS),
    reason='requires torch.load(weights_only=...) with safe_globals support',
)
def test_resume_checkpoint_default_weights_only_namespace(tmp_path):
    src_model = torch.nn.Linear(4, 2)
    src_optimizer = torch.optim.SGD(src_model.parameters(), lr=0.123, momentum=0.9)
    x = torch.randn(3, 4)
    src_optimizer.zero_grad()
    src_model(x).sum().backward()
    src_optimizer.step()

    checkpoint_path = tmp_path / 'resume_namespace_ckpt.pth'
    checkpoint = {
        'state_dict': src_model.state_dict(),
        'optimizer': src_optimizer.state_dict(),
        'epoch': 7,
        'version': 2,
        'args': argparse.Namespace(model='test-model'),
    }
    torch.save(checkpoint, checkpoint_path)

    dst_model = torch.nn.Linear(4, 2)
    dst_optimizer = torch.optim.SGD(dst_model.parameters(), lr=0.5, momentum=0.9)
    resume_epoch = resume_checkpoint(dst_model, checkpoint_path, optimizer=dst_optimizer, log_info=False)

    assert resume_epoch == 8
    assert torch.equal(dst_model.weight, src_model.weight)
    assert torch.equal(dst_model.bias, src_model.bias)
    assert dst_optimizer.param_groups[0]['lr'] == pytest.approx(0.123)
    assert len(dst_optimizer.state_dict()['state']) > 0


@pytest.mark.skipif(not _HAS_WEIGHTS_ONLY, reason='requires torch.load(weights_only=...) support')
def test_resume_checkpoint_blocks_non_allowlisted_globals(tmp_path):
    model = torch.nn.Linear(4, 2)
    checkpoint_path = tmp_path / 'resume_custom_ckpt.pth'
    checkpoint = {
        'state_dict': model.state_dict(),
        'args': _CustomPayload(11),
    }
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match='No automatic unsafe pickle fallback is performed'):
        resume_checkpoint(model, checkpoint_path, log_info=False)


def test_resume_checkpoint_weights_only_false_allows_custom_globals(tmp_path):
    src_model = torch.nn.Linear(4, 2)
    checkpoint_path = tmp_path / 'resume_custom_ckpt_unsafe.pth'
    checkpoint = {
        'state_dict': src_model.state_dict(),
        'epoch': 3,
        'version': 2,
        'args': _CustomPayload(11),
    }
    torch.save(checkpoint, checkpoint_path)

    dst_model = torch.nn.Linear(4, 2)
    resume_epoch = resume_checkpoint(dst_model, checkpoint_path, log_info=False, weights_only=False)

    assert resume_epoch == 4
    assert torch.equal(dst_model.weight, src_model.weight)
    assert torch.equal(dst_model.bias, src_model.bias)
