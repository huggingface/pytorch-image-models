import argparse
import inspect

import numpy as np
import pytest
import torch

from timm.models._helpers import load_state_dict, resume_checkpoint


_HAS_WEIGHTS_ONLY = 'weights_only' in inspect.signature(torch.load).parameters
_HAS_SAFE_GLOBALS = hasattr(torch.serialization, 'safe_globals')


class _CustomPayload:
    def __init__(self, value: int = 1):
        self.value = value


def _create_tiny_naflexvit(pretrained: bool = False, **kwargs):
    import timm

    return timm.create_model(
        'naflexvit_base_patch16_siglip',
        pretrained=pretrained,
        patch_size=2,
        embed_dim=4,
        depth=1,
        num_heads=1,
        mlp_ratio=2.0,
        pos_embed_grid_size=(2, 2),
        num_classes=0,
        fix_init=False,
        **kwargs,
    )


def _write_tiny_naflex_big_vision_npz(checkpoint_path, conv_patch_embed: bool = False):
    rng = np.random.default_rng(123)
    width = 4
    depth = 1
    num_heads = 1
    head_dim = width // num_heads
    mlp_width = 8
    patch_dim = 2 * 2 * 3

    def rand(shape):
        return rng.standard_normal(shape).astype(np.float32)

    block_prefix = 'params/img/Transformer/encoderblock/'
    block_attn_prefix = block_prefix + 'MultiHeadDotProductAttention_0/'
    pool_prefix = 'params/img/MAPHead_0/'
    pool_attn_prefix = pool_prefix + 'MultiHeadDotProductAttention_0/'
    weights = {
        'params/img/embedding/kernel': rand((patch_dim, width)),
        'params/img/embedding/bias': rand((width,)),
        'params/img/pos_embedding': rand((2, 2, width)),
        'params/img/Transformer/encoder_norm/scale': rand((width,)),
        'params/img/Transformer/encoder_norm/bias': rand((width,)),
        block_prefix + 'LayerNorm_0/scale': rand((depth, width)),
        block_prefix + 'LayerNorm_0/bias': rand((depth, width)),
        block_prefix + 'LayerNorm_1/scale': rand((depth, width)),
        block_prefix + 'LayerNorm_1/bias': rand((depth, width)),
        block_prefix + 'MlpBlock_0/Dense_0/kernel': rand((depth, width, mlp_width)),
        block_prefix + 'MlpBlock_0/Dense_0/bias': rand((depth, mlp_width)),
        block_prefix + 'MlpBlock_0/Dense_1/kernel': rand((depth, mlp_width, width)),
        block_prefix + 'MlpBlock_0/Dense_1/bias': rand((depth, width)),
        pool_prefix + 'probe': rand((1, 1, width)),
        pool_prefix + 'LayerNorm_0/scale': rand((width,)),
        pool_prefix + 'LayerNorm_0/bias': rand((width,)),
        pool_prefix + 'MlpBlock_0/Dense_0/kernel': rand((width, mlp_width)),
        pool_prefix + 'MlpBlock_0/Dense_0/bias': rand((mlp_width,)),
        pool_prefix + 'MlpBlock_0/Dense_1/kernel': rand((mlp_width, width)),
        pool_prefix + 'MlpBlock_0/Dense_1/bias': rand((width,)),
    }
    for prefix, kernel_shape, bias_shape in (
        (block_attn_prefix, (depth, width, num_heads, head_dim), (depth, num_heads, head_dim)),
        (pool_attn_prefix, (width, num_heads, head_dim), (num_heads, head_dim)),
    ):
        for name in ('query', 'key', 'value'):
            weights[prefix + name + '/kernel'] = rand(kernel_shape)
            weights[prefix + name + '/bias'] = rand(bias_shape)
    weights[block_attn_prefix + 'out/kernel'] = rand((depth, num_heads, head_dim, width))
    weights[block_attn_prefix + 'out/bias'] = rand((depth, width))
    weights[pool_attn_prefix + 'out/kernel'] = rand((num_heads, head_dim, width))
    weights[pool_attn_prefix + 'out/bias'] = rand((width,))

    if conv_patch_embed:
        # Classic ViT Big Vision layout: HWIO convolution and flattened NLC positions.
        weights['params/img/embedding/kernel'] = weights['params/img/embedding/kernel'].reshape(2, 2, 3, width)
        weights['params/img/pos_embedding'] = weights['params/img/pos_embedding'].reshape(1, 4, width)

    np.savez(checkpoint_path, **weights)
    return weights


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


@pytest.mark.parametrize('conv_patch_embed', [False, True])
def test_naflexvit_load_pretrained_big_vision_npz(tmp_path, conv_patch_embed):
    checkpoint_path = tmp_path / 'naflex_siglip2.npz'
    weights = _write_tiny_naflex_big_vision_npz(checkpoint_path, conv_patch_embed=conv_patch_embed)
    model = _create_tiny_naflexvit()

    model.load_pretrained(str(checkpoint_path))

    block_prefix = 'params/img/Transformer/encoderblock/'
    block_attn_prefix = block_prefix + 'MultiHeadDotProductAttention_0/'
    pool_prefix = 'params/img/MAPHead_0/'
    pool_attn_prefix = pool_prefix + 'MultiHeadDotProductAttention_0/'
    expected_qkv_weight = torch.cat([
        torch.from_numpy(weights[block_attn_prefix + name + '/kernel'][0]).flatten(1).T
        for name in ('query', 'key', 'value')
    ])
    expected_pool_kv_weight = torch.cat([
        torch.from_numpy(weights[pool_attn_prefix + name + '/kernel']).flatten(1).T for name in ('key', 'value')
    ])

    patch_embed_weight = weights['params/img/embedding/kernel']
    if patch_embed_weight.ndim == 2:
        expected_patch_embed_weight = torch.from_numpy(patch_embed_weight.T)
    else:
        expected_patch_embed_weight = torch.from_numpy(patch_embed_weight).permute(3, 0, 1, 2).flatten(1)
    pos_embed = weights['params/img/pos_embedding']
    if pos_embed.shape[0] == 1:
        pos_embed = pos_embed.reshape(1, 2, 2, pos_embed.shape[-1])
    else:
        pos_embed = pos_embed[None]

    assert torch.equal(model.embeds.proj.weight, expected_patch_embed_weight)
    assert torch.equal(model.embeds.proj.bias, torch.from_numpy(weights['params/img/embedding/bias']))
    assert torch.equal(model.embeds.pos_embed, torch.from_numpy(pos_embed))
    assert torch.equal(model.blocks[0].attn.qkv.weight, expected_qkv_weight)
    assert torch.equal(
        model.blocks[0].mlp.fc2.weight,
        torch.from_numpy(weights[block_prefix + 'MlpBlock_0/Dense_1/kernel'][0].T),
    )
    assert torch.equal(model.norm.weight, torch.from_numpy(weights['params/img/Transformer/encoder_norm/scale']))
    assert torch.equal(model.attn_pool.latent, torch.from_numpy(weights[pool_prefix + 'probe']))
    assert torch.equal(model.attn_pool.kv.weight, expected_pool_kv_weight)
    assert torch.equal(
        model.attn_pool.mlp.fc2.weight,
        torch.from_numpy(weights[pool_prefix + 'MlpBlock_0/Dense_1/kernel'].T),
    )

    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 4, 4))
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize('checkpoint_type', ['pth', 'safetensors', 'unrelated_npz'])
def test_naflexvit_load_pretrained_rejects_unsupported_checkpoint(tmp_path, checkpoint_type):
    model = _create_tiny_naflexvit()
    if checkpoint_type == 'pth':
        checkpoint_path = tmp_path / 'native.pth'
        torch.save(model.state_dict(), checkpoint_path)
    elif checkpoint_type == 'safetensors':
        checkpoint_path = tmp_path / 'native.safetensors'
        checkpoint_path.write_bytes((2).to_bytes(8, byteorder='little') + b'{}')
    else:
        checkpoint_path = tmp_path / 'unrelated.npz'
        np.savez(checkpoint_path, unrelated=np.ones(1, dtype=np.float32))

    with pytest.raises(ValueError, match=r'JAX/Flax.*factory/checkpoint loader'):
        model.load_pretrained(str(checkpoint_path))


def test_naflexvit_load_pretrained_disallows_pickled_arrays(tmp_path):
    checkpoint_path = tmp_path / 'object_array.npz'
    np.savez(
        checkpoint_path,
        **{'params/img/embedding/kernel': np.array([{'unsafe': True}], dtype=object)},
    )
    model = _create_tiny_naflexvit()

    with pytest.raises(ValueError, match='allow_pickle=False'):
        model.load_pretrained(str(checkpoint_path))


def test_naflexvit_native_checkpoint_uses_factory_loader(tmp_path):
    src_model = _create_tiny_naflexvit()
    checkpoint_path = tmp_path / 'naflex_native.pth'
    torch.save(src_model.state_dict(), checkpoint_path)

    dst_model = _create_tiny_naflexvit(
        pretrained=True,
        pretrained_cfg_overlay={'file': str(checkpoint_path)},
    )

    for key, value in src_model.state_dict().items():
        assert torch.equal(dst_model.state_dict()[key], value), key


def test_naflexvit_factory_remaps_classic_vit_checkpoint(tmp_path):
    import timm
    from timm.models.naflexvit import NaFlexVit

    model_kwargs = dict(
        img_size=4,
        patch_size=2,
        embed_dim=4,
        depth=1,
        num_heads=1,
        mlp_ratio=2.0,
        num_classes=3,
    )
    src_model = timm.create_model('vit_tiny_patch16_224', pretrained=False, **model_kwargs)
    checkpoint_path = tmp_path / 'classic_vit_native.pth'
    torch.save(src_model.state_dict(), checkpoint_path)

    dst_model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=True,
        use_naflex=True,
        pretrained_cfg_overlay={
            'file': str(checkpoint_path),
            'custom_load': False,
            'num_classes': model_kwargs['num_classes'],
        },
        **model_kwargs,
    )

    assert isinstance(dst_model, NaFlexVit)
    assert torch.equal(
        dst_model.embeds.proj.weight,
        src_model.patch_embed.proj.weight.permute(0, 2, 3, 1).flatten(1),
    )
    assert torch.equal(dst_model.embeds.proj.bias, src_model.patch_embed.proj.bias)
    assert torch.equal(
        dst_model.embeds.pos_embed,
        src_model.pos_embed[:, 1:].reshape_as(dst_model.embeds.pos_embed),
    )
    assert torch.equal(
        dst_model.embeds.cls_token,
        src_model.cls_token + src_model.pos_embed[:, :1],
    )

    remapped_keys = {
        'cls_token',
        'pos_embed',
        'patch_embed.proj.weight',
        'patch_embed.proj.bias',
    }
    dst_state_dict = dst_model.state_dict()
    for key, value in src_model.state_dict().items():
        if key not in remapped_keys:
            assert torch.equal(dst_state_dict[key], value), key

    src_model.eval()
    dst_model.eval()
    inputs = torch.randn(2, 3, 4, 4)
    with torch.inference_mode():
        expected = src_model(inputs)
        actual = dst_model(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
