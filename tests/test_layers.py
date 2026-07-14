import pytest
import torch
import torch.nn as nn

from timm.layers import (
    Attention2d,
    MultiQueryAttentionV2,
    PatchEmbedInterpolator,
    create_act_layer,
    get_act_fn,
    get_act_layer,
    resample_patch_embed,
    set_layer_config,
)

import importlib
import os

torch_backend = os.environ.get('TORCH_BACKEND')
if torch_backend is not None:
    importlib.import_module(torch_backend)
torch_device = os.environ.get('TORCH_DEVICE', 'cpu')


def test_patch_embed_interpolator_reuses_nonpersistent_cache(monkeypatch):
    interpolator = PatchEmbedInterpolator((4, 4), in_chans=3, embed_dim=8)
    linear_weight = torch.randn(8, 4 * 4 * 3, requires_grad=True)
    conv_weight = torch.randn(8, 3, 4, 4)
    expected_conv_weight = resample_patch_embed(conv_weight, [2, 2])

    original_pinv = torch.linalg.pinv
    pinv_calls = 0

    def counted_pinv(*args, **kwargs):
        nonlocal pinv_calls
        pinv_calls += 1
        return original_pinv(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, 'pinv', counted_pinv)
    first = interpolator.resample_linear_weight(linear_weight, (2, 2))
    second = interpolator.resample_linear_weight(linear_weight, (2, 2))
    actual_conv_weight = interpolator.resample_conv_weight(conv_weight, (2, 2))

    assert pinv_calls == 1
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(actual_conv_weight, expected_conv_weight)
    first.sum().backward()
    assert linear_weight.grad is not None
    assert (2, 2) in interpolator.resampler._pinv_cache_map
    assert not any(name.endswith('pinv_2x2') for name, _ in interpolator.named_buffers())
    assert not any('pinv_' in name for name in interpolator.state_dict())
    interpolator.to(dtype=torch.float16)
    assert not interpolator.resampler._pinv_cache_map


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA autocast')
def test_patch_embed_interpolator_cache_survives_autocast(monkeypatch):
    device = torch.device('cuda')
    interpolator = PatchEmbedInterpolator(
        (4, 4),
        in_chans=3,
        embed_dim=8,
        device=device,
    ).to(device)
    weight = torch.randn(8, 4 * 4 * 3, device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        first = interpolator.resample_linear_weight(weight, (2, 2))

    cached_matrix = interpolator.resampler._pinv_cache_map[(2, 2)]
    assert cached_matrix.dtype == torch.float32

    def unexpected_pinv(*args, **kwargs):
        raise AssertionError('cache hit recomputed the pseudoinverse')

    monkeypatch.setattr(torch.linalg, 'pinv', unexpected_pinv)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        second = interpolator.resample_linear_weight(weight, (2, 2))

    torch.testing.assert_close(first, second)


@pytest.mark.skipif(not hasattr(torch, 'compile'), reason='requires torch.compile')
def test_patch_embed_interpolator_prewarm_supports_fullgraph_compile():
    class InterpolatingProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.interpolator = PatchEmbedInterpolator((4, 4), in_chans=3, embed_dim=8)
            self.weight = nn.Parameter(torch.randn(8, 4 * 4 * 3))
            self.interpolator.prewarm([(2, 2)])

        def forward(self, x):
            return self.interpolator(
                x,
                self.weight,
                patch_size=(2, 2),
                is_linear=True,
            )

    module = InterpolatingProjection().eval()
    patches = torch.randn(1, 3, 2, 2, 3)

    with torch.inference_mode():
        expected = module(patches)
        actual = torch.compile(module, backend='eager', fullgraph=True)(patches)

    torch.testing.assert_close(actual, expected)

class MLP(nn.Module):
    def __init__(self, act_layer="relu", inplace=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.act = create_act_layer(act_layer, inplace=inplace)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def _run_act_layer_grad(act_type, inplace=True):
    x = torch.rand(10, 1000) * 10
    m = MLP(act_layer=act_type, inplace=inplace)

    def _run(x, act_layer=''):
        if act_layer:
            # replace act layer if set
            m.act = create_act_layer(act_layer, inplace=inplace)
        out = m(x)
        l = (out - 0).pow(2).sum()
        return l

    x = x.to(device=torch_device)
    m.to(device=torch_device)

    out_me = _run(x)

    with set_layer_config(scriptable=True):
        out_jit = _run(x, act_type)

    assert torch.isclose(out_jit, out_me)

    with set_layer_config(no_jit=True):
        out_basic = _run(x, act_type)

    assert torch.isclose(out_basic, out_jit)


def test_swish_grad():
    for _ in range(100):
        _run_act_layer_grad('swish')


def test_mish_grad():
    for _ in range(100):
        _run_act_layer_grad('mish')


def test_hard_sigmoid_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_sigmoid', inplace=None)


def test_hard_swish_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_swish')


def test_hard_mish_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_mish')

def test_get_act_layer_empty_string():
    # Empty string should return None
    assert get_act_layer('') is None


def test_create_act_layer_inplace_error():
    class NoInplaceAct(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x
    
    # Should recover when inplace arg causes TypeError
    layer = create_act_layer(NoInplaceAct, inplace=True)
    assert isinstance(layer, NoInplaceAct)


def test_create_act_layer_edge_cases():
    # Test None input
    assert create_act_layer(None) is None
    
    # Test TypeError handling for inplace
    class CustomAct(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, x):
            return x
            
    result = create_act_layer(CustomAct, inplace=True)
    assert isinstance(result, CustomAct)


def test_get_act_fn_callable():
    def custom_act(x): 
        return x
    assert get_act_fn(custom_act) is custom_act


def test_get_act_fn_none():
    assert get_act_fn(None) is None
    assert get_act_fn('') is None


@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("dim_out", [128, 256])
@pytest.mark.parametrize("use_m", [True, False])
def test_mqa_v2(dim, dim_out, use_m):
    mqa = MultiQueryAttentionV2(dim, dim_out)
    
    x = torch.randn(1, dim, 32, 48)
    if use_m:
        m = torch.randn(1, dim, 16, 24)
    else:
        m = None
        
    y = mqa(x, m=m)
    
    assert (y.shape) == (1, dim_out, 32, 48)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("expand_first", [True, False])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("attn_mask", [True, False])
def test_attn2d(bias, expand_first, head_first, attn_mask):
    x = torch.randn(1, 128, 32, 48)
    attn = Attention2d(
        128, 128, num_heads=4, bias=bias, expand_first=expand_first, head_first=head_first
    )
    
    if attn_mask:
        mask = torch.randint(0, 1, size=(32 * 48, 32 * 48), dtype=torch.float32)
    else:
        mask = None
    
    o1 = attn(x, mask)
    attn.fused_attn = False
    o2 = attn(x, mask)
    
    assert torch.allclose(o1, o2, atol=1e-5), f"{torch.abs(o1 - o2).max()}"
