"""Tests for timm/layers/pos_embed_sincos.py

The public builder functions (build_fourier_pos_embed, build_rotary_pos_embed) are pinned bitwise against verbatim
copies of their pre-refactor implementations, the modules are pinned against those functions. The final section
covers train-time coordinate augmentation (reference numerics, RNG consumption, caches, model integration).
"""
import math
from itertools import product

import pytest
import torch

from timm.layers import (
    RotaryEmbedding,
    RotaryEmbeddingCat,
    RotaryEmbeddingDinoV3,
    RotaryEmbeddingMixed,
    RotaryEmbeddingMRope,
    apply_rot_embed,
    apply_rot_embed_cat,
    apply_rot_embed_list,
    build_fourier_pos_embed,
    build_rotary_pos_embed,
    freq_bands,
    pixel_freq_bands,
)
from timm.layers.pos_embed_sincos import (
    _build_grid, create_rope_embed, get_mixed_grid, make_coords_dinov3, swap_shape_xy,
)


# ---------------------------------------------------------------------------------------------------------------------
# Verbatim copies of the pre-refactor public builders (reference for bitwise compat).
# ---------------------------------------------------------------------------------------------------------------------

def _legacy_build_fourier_pos_embed(
        feat_shape,
        bands=None,
        num_bands=64,
        max_res=224,
        temperature=10000.,
        linear_bands=False,
        include_grid=False,
        in_pixels=True,
        ref_feat_shape=None,
        grid_offset=0.,
        grid_indexing='ij',
        device=None,
        dtype=torch.float32,
):
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
                device=device,
            )
        else:
            bands = freq_bands(
                num_bands,
                temperature=temperature,
                step=1,
                device=device,
            )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if grid_indexing == 'xy':
        feat_shape = swap_shape_xy(feat_shape)
        if ref_feat_shape is not None:
            ref_feat_shape = swap_shape_xy(ref_feat_shape)

    if in_pixels:
        t = [
            torch.linspace(-1., 1., steps=s, device=device, dtype=torch.float32)
            for s in feat_shape
        ]
    else:
        t = [
            torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) + grid_offset
            for s in feat_shape
        ]

    if ref_feat_shape is not None:
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = torch.stack(torch.meshgrid(t, indexing=grid_indexing), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype=dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


def _legacy_build_rotary_pos_embed(
        feat_shape,
        bands=None,
        dim=64,
        max_res=224,
        temperature=10000.,
        linear_bands=False,
        in_pixels=True,
        ref_feat_shape=None,
        grid_offset=0.,
        grid_indexing='ij',
        rotate_half=False,
        device=None,
        dtype=torch.float32,
):
    sin_emb, cos_emb = _legacy_build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape(num_spatial_dim, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1)
    if rotate_half:
        sin_emb = torch.cat([sin_emb, sin_emb], dim=-1)
        cos_emb = torch.cat([cos_emb, cos_emb], dim=-1)
    else:
        sin_emb = sin_emb.repeat_interleave(2, -1)
        cos_emb = cos_emb.repeat_interleave(2, -1)
    return sin_emb, cos_emb


def _assert_all_equal(a, b):
    a = a if isinstance(a, (list, tuple)) else [a]
    b = b if isinstance(b, (list, tuple)) else [b]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert torch.equal(x, y)


def _legacy_dinov3_coords(shape, normalize_coords='separate', grid_offset=0., device='cpu'):
    h, w = shape
    h_denom, w_denom = float(h), float(w)
    if normalize_coords == 'min':
        h_denom = w_denom = float(min(h, w))
    elif normalize_coords == 'max':
        h_denom = w_denom = float(max(h, w))
    h_coords = (torch.arange(0.5, h, device=device) + grid_offset) / h_denom
    w_coords = (torch.arange(0.5, w, device=device) + grid_offset) / w_denom
    return 2.0 * torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), -1).flatten(0, 1) - 1.0


def _legacy_mixed_embed(freqs, t_x, t_y):
    output_dtype = freqs.dtype
    freqs = freqs.float()
    angles = t_x.float().unsqueeze(-1) @ freqs[0].unsqueeze(-2)
    angles = angles + (t_y.float().unsqueeze(-1) @ freqs[1].unsqueeze(-2))
    sin = angles.sin().repeat_interleave(2, -1)
    cos = angles.cos().repeat_interleave(2, -1)
    return torch.cat([sin, cos], -1).to(output_dtype)


def _legacy_mrope_embed(shape, dim, sections, temperature, device):
    # Pin the original channel assignments and the expand-before-trig order independently of module helpers.
    h, w = shape
    inv_freq = 1.0 / (temperature ** (torch.arange(0, dim, 2, device=device).float() / dim))
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    pos = torch.zeros(h * w, dim // 2, device=device)
    pos[:, 1:sections[1] * 3:3] = ys.flatten().float()[:, None]
    pos[:, 2:sections[2] * 3:3] = xs.flatten().float()[:, None]
    angles = pos * inv_freq
    angles = torch.cat([angles, angles], -1)
    return torch.cat([angles.sin(), angles.cos()], -1)


@pytest.fixture(params=['cpu', 'cuda'])
def rope_device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return torch.device(request.param)


_GRID_CASES = list(product(
    [(8, 8), (5, 9), (3, 4, 5)],  # feat_shape
    [False, True],  # in_pixels
    [None, 'ref'],  # ref_feat_shape (same ndim as feat_shape)
    [0., 0.5],  # grid_offset
    ['ij', 'xy'],  # grid_indexing
))


def _ref_shape(feat_shape, ref):
    return None if ref is None else tuple(2 * s for s in feat_shape)


@pytest.mark.parametrize('feat_shape,in_pixels,ref,grid_offset,grid_indexing', _GRID_CASES)
@pytest.mark.parametrize('include_grid', [False, True])
@pytest.mark.parametrize('bands_dtype', [None, torch.float32, torch.bfloat16])
def test_build_fourier_pos_embed_matches_legacy(
        feat_shape, in_pixels, ref, grid_offset, grid_indexing, include_grid, bands_dtype,
):
    kwargs = dict(
        num_bands=8, max_res=64, temperature=100., linear_bands=in_pixels, include_grid=include_grid,
        in_pixels=in_pixels, ref_feat_shape=_ref_shape(feat_shape, ref), grid_offset=grid_offset,
        grid_indexing=grid_indexing,
    )
    dtypes = [torch.float32, torch.bfloat16]
    if bands_dtype is not None:
        # pre-computed bands path, incl. the dtype=None -> bands.dtype behaviour
        kwargs['bands'] = pixel_freq_bands(8, 64.) if in_pixels else freq_bands(8, 100., step=1)
        kwargs['bands'] = kwargs['bands'].to(bands_dtype)
        dtypes.append(None)
    for dtype in dtypes:
        _assert_all_equal(
            build_fourier_pos_embed(feat_shape, dtype=dtype, **kwargs),
            _legacy_build_fourier_pos_embed(feat_shape, dtype=dtype, **kwargs),
        )


@pytest.mark.parametrize('feat_shape,in_pixels,ref,grid_offset,grid_indexing', _GRID_CASES)
@pytest.mark.parametrize('rotate_half', [False, True])
@pytest.mark.parametrize('use_bands', [False, True])
def test_build_rotary_pos_embed_matches_legacy(
        feat_shape, in_pixels, ref, grid_offset, grid_indexing, rotate_half, use_bands,
):
    kwargs = dict(
        dim=32, max_res=64, temperature=100., linear_bands=in_pixels, in_pixels=in_pixels,
        ref_feat_shape=_ref_shape(feat_shape, ref), grid_offset=grid_offset, grid_indexing=grid_indexing,
        rotate_half=rotate_half,
    )
    if use_bands:
        kwargs['bands'] = pixel_freq_bands(8, 64.) if in_pixels else freq_bands(8, 100., step=1)
    for dtype in (torch.float32, torch.float16):
        _assert_all_equal(
            build_rotary_pos_embed(feat_shape, dtype=dtype, **kwargs),
            _legacy_build_rotary_pos_embed(feat_shape, dtype=dtype, **kwargs),
        )


@pytest.mark.parametrize('normalize_coords', ['separate', 'min', 'max'])
@pytest.mark.parametrize('grid_offset', [0., 0.25])
@pytest.mark.parametrize('feat_shape', [(8, 8), (6, 9)])
@pytest.mark.parametrize('grid_indexing', ['ij', 'xy'])
def test_build_grid_centered_matches_dinov3_coords(normalize_coords, grid_offset, feat_shape, grid_indexing):
    grid = _build_grid(
        feat_shape, grid_type='centered', grid_offset=grid_offset,
        normalize_coords=normalize_coords, grid_indexing=grid_indexing,
    )
    assert grid.shape == (*feat_shape, 2)
    coords = make_coords_dinov3(
        *feat_shape, normalize_coords=normalize_coords, grid_offset=grid_offset, grid_indexing=grid_indexing,
    )
    _assert_all_equal(grid.flatten(0, 1), coords)
    legacy = _legacy_dinov3_coords(feat_shape, normalize_coords, grid_offset)
    _assert_all_equal(coords, legacy if grid_indexing == 'ij' else legacy.flip(-1))


def test_build_grid_xy_swaps_channels():
    grid_ij = _build_grid((4, 6), grid_type='index', grid_indexing='ij')
    grid_xy = _build_grid((4, 6), grid_type='index', grid_indexing='xy')
    assert grid_ij.shape == grid_xy.shape == (4, 6, 2)
    _assert_all_equal(grid_xy.flip(-1), grid_ij)
    with pytest.raises(ValueError):
        _build_grid((4, 6), grid_type='unknown')


# ---------------------------------------------------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------------------------------------------------

_MODULE_KWARGS = [
    dict(in_pixels=False),
    dict(in_pixels=False, ref_feat_shape=(12, 12), grid_offset=0.5, grid_indexing='xy'),
    dict(in_pixels=True, linear_bands=True),
    dict(in_pixels=False, grid_type='centered', normalize_coords='min'),
]


@pytest.mark.parametrize('cls', [RotaryEmbedding, RotaryEmbeddingCat])
@pytest.mark.parametrize('kwargs', _MODULE_KWARGS)
@pytest.mark.parametrize('rotate_half', [False, True])
def test_rotary_module_matches_functional(cls, kwargs, rotate_half):
    kwargs = dict(kwargs, rotate_half=rotate_half)
    fn_kwargs = {k: v for k, v in kwargs.items() if k not in ('grid_type', 'normalize_coords')}
    grid_kwargs = {k: v for k, v in kwargs.items() if k in ('grid_type', 'normalize_coords')}

    def expected(shape):
        out = build_rotary_pos_embed(shape, dim=64, **fn_kwargs, **grid_kwargs)
        return torch.cat(out, -1) if cls is RotaryEmbeddingCat else out

    if 'grid_type' not in kwargs:
        # no new args -> the module must also match the legacy functional impl
        out_legacy = _legacy_build_rotary_pos_embed((8, 8), dim=64, **fn_kwargs)
        _assert_all_equal(out_legacy, build_rotary_pos_embed((8, 8), dim=64, **fn_kwargs))

    m_dyn = create_rope_embed('base' if cls is RotaryEmbedding else 'cat', dim=128, num_heads=2, **kwargs)
    _assert_all_equal(m_dyn.get_embed((8, 8)), expected((8, 8)))
    _assert_all_equal(m_dyn.get_embed((5, 9)), expected((5, 9)))

    m_cache = cls(64, feat_shape=(8, 8), **kwargs)
    _assert_all_equal(m_cache.get_embed(), expected((8, 8)))
    _assert_all_equal(m_cache.get_embed((5, 9)), expected((8, 8)))  # shape ignored in cached mode
    m_cache.update_feat_shape((5, 9))
    _assert_all_equal(m_cache.get_embed(), expected((5, 9)))


@pytest.mark.parametrize('kwargs', _MODULE_KWARGS)
@pytest.mark.parametrize('seq_len', [None, 100])
def test_rotary_cat_get_batch_embeds_matches_get_embed(kwargs, seq_len):
    m = RotaryEmbeddingCat(64, feat_shape=None, **kwargs)
    shapes = [(8, 8), (5, 9), (9, 5), (8, 8)]
    out = m.get_batch_embeds(shapes, seq_len=seq_len)
    if seq_len is None:
        assert isinstance(out, list) and len(out) == len(shapes)
        for shape, embed in zip(shapes, out):
            _assert_all_equal(embed, m.get_embed(shape))
    else:
        assert out.shape == (len(shapes), seq_len, 128)
        for i, shape in enumerate(shapes):
            n = shape[0] * shape[1]
            _assert_all_equal(out[i, :n], m.get_embed(shape))
            assert not out[i, n:].any()
    assert m.get_batch_embeds([]) == []
    with pytest.raises(RuntimeError):
        RotaryEmbeddingCat(64, feat_shape=(8, 8), **kwargs).get_batch_embeds(shapes)


@pytest.mark.parametrize('rotate_half', [True, False])
@pytest.mark.parametrize('normalize_coords', ['separate', 'max'])
def test_dinov3_module(rotate_half, normalize_coords):
    kwargs = dict(rotate_half=rotate_half, normalize_coords=normalize_coords, grid_offset=0.25)
    m = RotaryEmbeddingDinoV3(64, feat_shape=None, **kwargs).eval()

    # Independent reference preserves the pretrained 'ij' coordinate order and expand-before-trig numerics.
    def expected(shape):
        coords = _legacy_dinov3_coords(shape, normalize_coords=normalize_coords, grid_offset=0.25)
        angles = 2 * math.pi * coords[:, :, None] / m.periods[None, None, :]
        angles = angles.flatten(1)
        angles = angles.tile(2) if rotate_half else angles.repeat_interleave(2, dim=-1)
        return torch.cat([angles.sin(), angles.cos()], -1)

    _assert_all_equal(m.get_embed((8, 8)), expected((8, 8)))
    _assert_all_equal(m.get_embed((5, 9)), expected((5, 9)))

    m_cache = RotaryEmbeddingDinoV3(64, feat_shape=(8, 8), **kwargs).eval()
    _assert_all_equal(m_cache.get_embed(), expected((8, 8)))
    m_cache.update_feat_shape((5, 9))
    _assert_all_equal(m_cache.get_embed(), expected((5, 9)))

    shapes = [(8, 8), (5, 9), (8, 8)]
    for shape, embed in zip(shapes, m.get_batch_embeds(shapes)):
        _assert_all_equal(embed, expected(shape))
    padded = m.get_batch_embeds(shapes, seq_len=80)
    assert padded.shape == (3, 80, 128)
    _assert_all_equal(padded[1, :45], expected((5, 9)))
    assert not padded[1, 45:].any()


@pytest.mark.parametrize('cls', [RotaryEmbedding, RotaryEmbeddingCat, RotaryEmbeddingDinoV3])
@pytest.mark.parametrize('low_dtype', [torch.bfloat16, torch.float16])
def test_rotary_freqs_stay_fp32_after_cast(cls, low_dtype):
    def freqs(m):
        return m.periods if cls is RotaryEmbeddingDinoV3 else m.bands

    def cat(x):
        return torch.cat(x, -1) if isinstance(x, (tuple, list)) else x

    ref = cls(64, feat_shape=None)
    ref_out = cat(ref.get_embed((16, 16)))
    assert ref_out.dtype == torch.float32

    # cast after construction: frequencies rebuilt in float32, output identical to the float32 module
    m = cls(64, feat_shape=None).to(low_dtype)
    assert freqs(m).dtype == torch.float32
    _assert_all_equal(cat(m.get_embed((16, 16))), ref_out)
    _assert_all_equal(cat(m.get_embed((16, 16), dtype=low_dtype)), ref_out.to(low_dtype))
    # Repeated casts keep frequency buffers in float32.
    assert freqs(m.to(torch.float32).to(low_dtype).cpu()).dtype == torch.float32

    # low precision dtype at construction: frequency buffer still float32
    m = cls(64, feat_shape=None, dtype=low_dtype)
    assert freqs(m).dtype == torch.float32
    _assert_all_equal(cat(m.get_embed((16, 16))), ref_out)

    # float64 is not downcast
    assert freqs(cls(64, feat_shape=None, dtype=torch.float64)).dtype == torch.float64

    # cached embeds follow the model dtype (cast once, at the end)
    m = cls(64, feat_shape=(16, 16)).to(low_dtype)
    _assert_all_equal(cat(m.get_embed()), ref_out.to(low_dtype))
    _assert_all_equal(cat(m.get_embed(dtype=torch.float32)), ref_out.to(low_dtype).float())


class _GetEmbedWrapper(torch.nn.Module):
    """Wrap get_embed(shape) in a forward for FX tracing."""

    def __init__(self, rope):
        super().__init__()
        self.rope = rope

    def forward(self, x):
        return self.rope.get_embed(shape=[x.shape[2], x.shape[3]])


_FX_CASES = [
    lambda: RotaryEmbeddingCat(64, in_pixels=False, feat_shape=None),
    lambda: RotaryEmbeddingCat(64, in_pixels=False, feat_shape=None, ref_feat_shape=(8, 8), rotate_half=True),
    lambda: RotaryEmbeddingCat(64, in_pixels=False, feat_shape=None, grid_type='centered', grid_indexing='xy'),
    lambda: RotaryEmbeddingCat(64, in_pixels=False, feat_shape=(8, 8)),
    lambda: RotaryEmbedding(64, in_pixels=True, feat_shape=None),
    lambda: RotaryEmbeddingDinoV3(64, feat_shape=None, normalize_coords='max'),
    lambda: RotaryEmbeddingDinoV3(64, feat_shape=(8, 8)),
    lambda: RotaryEmbeddingMixed(64, depth=2, num_heads=2),
    lambda: RotaryEmbeddingMixed(64, depth=2, num_heads=2, feat_shape=(8, 8)),
    lambda: RotaryEmbeddingMRope(64),
]


@pytest.mark.parametrize('make', _FX_CASES)
def test_rotary_fx(make):
    m = _GetEmbedWrapper(make().eval())
    traced = torch.fx.symbolic_trace(m)
    for shape in ((8, 8), (5, 9)):
        x = torch.zeros(1, 1, *shape)
        _assert_all_equal(traced(x), m(x))


@pytest.mark.parametrize('cls', [RotaryEmbedding, RotaryEmbeddingCat, RotaryEmbeddingDinoV3])
@pytest.mark.parametrize('feat_shape', [None, (8, 8)])
def test_rotary_init_after_to_empty(cls, feat_shape):
    ref = cls(64, feat_shape=feat_shape)
    m = cls(64, feat_shape=feat_shape, device='meta')
    assert all(b.device.type == 'meta' for b in m.buffers())
    m = m.to_empty(device='cpu')
    for b in m.buffers():
        b.fill_(float('nan'))
    m.init_non_persistent_buffers()
    for (n, b), (_, rb) in zip(m.named_buffers(), ref.named_buffers()):
        _assert_all_equal(b, rb)
    if feat_shape is None:
        _assert_all_equal(m.get_embed((8, 8)), ref.get_embed((8, 8)))
    else:
        _assert_all_equal(m.get_embed(), ref.get_embed())


@pytest.mark.parametrize('cls', [RotaryEmbedding, RotaryEmbeddingCat, RotaryEmbeddingDinoV3])
@pytest.mark.parametrize('feat_shape', [None, (8, 8)])
@pytest.mark.parametrize('rotate_half', [False, True])
def test_rotary_module_forward_nchw(cls, feat_shape, rotate_half):
    m = cls(64, feat_shape=feat_shape, rotate_half=rotate_half).eval()
    x = torch.randn(2, 64, 8, 8)
    out = m(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    embed = m.get_embed((8, 8))
    x_flat = x.flatten(2).transpose(1, 2)
    if cls is RotaryEmbedding:
        expected = apply_rot_embed(x_flat, *embed, half=rotate_half)
    else:
        expected = apply_rot_embed_cat(x_flat, embed, half=m.rotate_half)
    _assert_all_equal(out, expected.transpose(1, 2).reshape(x.shape))
    # low precision input keeps its dtype, math happens in float32 (cast once at the end)
    out_bf = m(x.to(torch.bfloat16))
    assert out_bf.dtype == torch.bfloat16
    torch.testing.assert_close(out_bf.float(), out, atol=2e-2, rtol=2e-2)
    if feat_shape is None:
        assert m(torch.randn(2, 64, 5, 9)).shape == (2, 64, 5, 9)
    _assert_all_equal(torch.fx.symbolic_trace(m)(x), out)


def test_mixed_rope_low_precision_cast():
    m_cache = RotaryEmbeddingMixed(64, depth=2, num_heads=2, feat_shape=(8, 8))
    m_dyn = RotaryEmbeddingMixed(64, depth=2, num_heads=2, feat_shape=None)
    with torch.no_grad():
        m_dyn.freqs.copy_(m_cache.freqs)
    ref = m_cache.get_embed()
    _assert_all_equal(m_dyn.get_embed((8, 8)), ref)

    m_cache = m_cache.to(torch.bfloat16)
    m_dyn = m_dyn.to(torch.bfloat16)
    assert m_cache.t_x.dtype == torch.float32 and m_cache.t_y.dtype == torch.float32
    assert m_cache.freqs.dtype == torch.bfloat16
    out_cache = m_cache.get_embed()
    out_dyn = m_dyn.get_embed((8, 8))
    assert out_cache.dtype == out_dyn.dtype == torch.bfloat16
    assert torch.isfinite(out_cache).all()
    _assert_all_equal(out_cache, out_dyn)  # cached grid buffers must match the on-the-fly grid path
    m_cache.update_feat_shape((5, 9))
    _assert_all_equal(m_cache.get_embed(), m_dyn.get_embed((5, 9)))


def test_dinov3_temperature_none():
    m = RotaryEmbeddingDinoV3(64, temperature=None, min_period=0.5, max_period=10., feat_shape=None)
    ref = RotaryEmbeddingDinoV3(64, temperature=100., min_period=0.5, max_period=10., feat_shape=None)
    assert m.temperature is None
    _assert_all_equal(m.periods, ref.periods)
    _assert_all_equal(m.get_embed((8, 8)), ref.get_embed((8, 8)))
    with pytest.raises(ValueError):
        RotaryEmbeddingDinoV3(64, temperature=None)


@pytest.mark.parametrize('grid_indexing', ['ij', 'xy'])
@pytest.mark.parametrize('shape', [(5, 9), (3, 513)])
def test_mixed_rope_float32_autocast_and_gradients(rope_device, grid_indexing, shape):
    torch.manual_seed(42)
    m = RotaryEmbeddingMixed(64, depth=2, num_heads=2, grid_indexing=grid_indexing, device=rope_device)
    # Reference grid construction predates the shared helper.
    axes = shape if grid_indexing == 'ij' else shape[::-1]
    grid = torch.meshgrid(*[torch.arange(s, device=rope_device).float() for s in axes], indexing=grid_indexing)
    t_x, t_y = [t.flatten() for t in grid]
    _assert_all_equal(get_mixed_grid(list(shape), grid_indexing=grid_indexing, device=rope_device), (t_x, t_y))
    expected = _legacy_mixed_embed(m.freqs, t_x, t_y)
    actual = m.get_embed(shape)
    _assert_all_equal(actual, expected)

    weights = torch.randn_like(expected) / (shape[0] * shape[1])
    grad_ref = torch.autograd.grad(expected, m.freqs, weights)[0]
    grad_actual = torch.autograd.grad(actual, m.freqs, weights)[0]
    # Matmul and broadcast multiplication reduce parameter gradients in different orders.
    torch.testing.assert_close(grad_actual, grad_ref, atol=1e-5, rtol=1e-4)

    amp_dtypes = [torch.bfloat16, torch.float16] if rope_device.type == 'cuda' else [torch.bfloat16]
    for dtype in amp_dtypes:
        with torch.autocast(rope_device.type, dtype=dtype):
            amp = m.get_embed(shape)
        _assert_all_equal(amp, actual)
        grad_amp = torch.autograd.grad(amp, m.freqs, weights)[0]
        _assert_all_equal(grad_amp, grad_actual)


@pytest.mark.parametrize('low_dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('dim,sections', [(64, (8, 12, 12)), (72, (12, 12, 12)), (80, (4, 20, 2))])
def test_mrope_reference_precision_and_meta_init(rope_device, low_dtype, dim, sections):
    shape = (3, 513)
    expected = _legacy_mrope_embed(shape, dim, sections, 10000., rope_device)
    m = RotaryEmbeddingMRope(dim, mrope_section=sections, device=rope_device)
    _assert_all_equal(m.get_embed(shape), expected)
    m.to(low_dtype)
    assert m.inv_freq.dtype == torch.float32
    _assert_all_equal(m.get_embed(shape), expected)
    _assert_all_equal(m.get_embed(shape, dtype=low_dtype), expected.to(low_dtype))
    constructed = RotaryEmbeddingMRope(dim, mrope_section=sections, dtype=low_dtype, device=rope_device)
    _assert_all_equal(constructed.get_embed(shape), expected)

    meta = RotaryEmbeddingMRope(dim, mrope_section=sections, dtype=low_dtype, device='meta')
    meta.to_empty(device=rope_device)
    meta.inv_freq.fill_(float('nan'))
    meta.axis.fill_(-1)
    meta.init_non_persistent_buffers()
    _assert_all_equal(meta.get_embed(shape), expected)
    _assert_all_equal(meta.axis, m.axis)
    assert not meta.state_dict()  # both buffers must remain non-persistent


@pytest.mark.parametrize('rotate_half', [False, True])
@pytest.mark.parametrize('grid_indexing', ['ij', 'xy'])
def test_dinov3_reference_and_cache_dtype(rope_device, rotate_half, grid_indexing):
    m = RotaryEmbeddingDinoV3(
        64, feat_shape=(8, 8), rotate_half=rotate_half, grid_indexing=grid_indexing,
        normalize_coords='min', grid_offset=0.25, device=rope_device,
    ).eval()
    shape = (5, 9)
    # timm builds and normalizes its grid on CPU; upstream DINOv3 builds on the periods device.
    coords = _legacy_dinov3_coords(shape, normalize_coords='min', grid_offset=0.25).to(rope_device)
    if grid_indexing == 'xy':
        coords = coords.flip(-1)
    angles = (2 * math.pi * coords[:, :, None] / m.periods[None, None, :]).flatten(1)
    angles = angles.tile(2) if rotate_half else angles.repeat_interleave(2, -1)
    expected = torch.cat([angles.sin(), angles.cos()], -1)
    _assert_all_equal(m.get_embed(shape), expected)
    m.bfloat16()
    m.update_feat_shape(shape)
    _assert_all_equal(m.get_embed(), expected.bfloat16())
    _assert_all_equal(m.get_embed(shape), expected)
    m.init_non_persistent_buffers()
    _assert_all_equal(m.get_embed(), expected.bfloat16())


@pytest.mark.parametrize('kind', ['cat', 'dinov3', 'mixed'])
def test_rope_batch_padding_dtype_and_gradients(kind):
    if kind == 'mixed':
        m = RotaryEmbeddingMixed(64, depth=2, num_heads=2).bfloat16()
    else:
        cls = RotaryEmbeddingCat if kind == 'cat' else RotaryEmbeddingDinoV3
        m = cls(64).eval()
    shapes = [(5, 9), (8, 4)]
    embeds = m.get_batch_embeds(shapes, dtype=torch.float32)
    padded = m.get_batch_embeds(shapes, seq_len=48, dtype=torch.float32)
    for i, shape in enumerate(shapes):
        n = shape[0] * shape[1]
        expected = m.get_embed(shape, dtype=torch.float32)
        _assert_all_equal(embeds[i], expected)
        _assert_all_equal(padded[i, ..., :n, :], expected)
        assert not padded[i, ..., n:, :].any()
    with pytest.raises(ValueError, match='seq_len'):
        m.get_batch_embeds(shapes, seq_len=10)
    assert m.get_batch_embeds([], seq_len=48) == []
    if kind == 'mixed':
        # An explicit float32 output must not take a round trip through the parameter's BF16 dtype.
        t_x, t_y = get_mixed_grid(list(shapes[0]), grid_indexing=m.grid_indexing)
        _assert_all_equal(embeds[0], _legacy_mixed_embed(m.freqs.float(), t_x, t_y))
        padded.mean().backward()
        assert m.freqs.grad is not None and torch.isfinite(m.freqs.grad).all() and m.freqs.grad.abs().sum() > 0


@pytest.mark.parametrize('feat_shape', [None, (5, 9)])
@pytest.mark.parametrize('block_index', [0, 1])
def test_mixed_rope_forward_per_block(feat_shape, block_index):
    torch.manual_seed(42)
    m = RotaryEmbeddingMixed(64, depth=2, num_heads=2, feat_shape=feat_shape)
    x = torch.randn(2, 64, 5, 9, requires_grad=True)
    expected_heads = x.reshape(2, 2, 32, 45).transpose(-2, -1)
    expected = apply_rot_embed_cat(expected_heads, m.get_embed((5, 9))[block_index])
    expected = expected.transpose(-2, -1).reshape_as(x)
    out = m(x, block_index=block_index)
    _assert_all_equal(out, expected)
    _assert_all_equal(torch.fx.symbolic_trace(m)(x, block_index), out)
    (out * torch.randn_like(out)).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.freqs.grad is not None and torch.isfinite(m.freqs.grad).all()
    assert m.freqs.grad[:, block_index].abs().sum() > 0
    assert not m.freqs.grad[:, 1 - block_index].any()
    assert m(x.bfloat16(), block_index).dtype == torch.bfloat16
    with pytest.raises(AssertionError, match='block_index'):
        m(x, block_index=2)


def _reference_rotary_apply(
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        half: bool,
) -> torch.Tensor:
    # Independent reference arithmetic before output dtype conversion.
    if half:
        first, second = x.chunk(2, -1)
        rotated = torch.cat([-second, first], -1)
    else:
        rotated = torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape_as(x)
    return x * cos + rotated * sin


@pytest.mark.parametrize('half', [False, True])
@pytest.mark.parametrize('input_dtype,embed_dtype', [
    (torch.bfloat16, torch.float32), (torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16),
    (torch.float32, torch.float32), (torch.float64, torch.float32), (torch.float32, torch.float64),
])
def test_rotary_apply_output_dtype_and_rounding(half, input_dtype, embed_dtype, rope_device):
    torch.manual_seed(123)
    x = torch.randn(2, 3, 45, 32, device=rope_device, dtype=input_dtype, requires_grad=True)
    sin, cos = build_rotary_pos_embed((5, 9), dim=32, rotate_half=half, device=rope_device, dtype=embed_dtype)
    sin.requires_grad_()
    cos.requires_grad_()
    # The reference retains the original arithmetic, rounding only the completed result.
    expected = _reference_rotary_apply(x, sin, cos, half).to(input_dtype)
    cat_embed = torch.cat([sin, cos], -1)
    outputs = [apply_rot_embed(x, sin, cos, half), apply_rot_embed_cat(x, cat_embed, half)]
    outputs += apply_rot_embed_list([x, x], sin, cos, half)
    outputs += apply_rot_embed_list(x, sin, cos, half)
    for out in outputs:
        assert out.dtype == input_dtype
        _assert_all_equal(out, expected)
    # Downcasting the embeddings first is a different result and is deliberately avoided.
    if input_dtype in (torch.bfloat16, torch.float16) and embed_dtype == torch.float32:
        early_cast = _reference_rotary_apply(x, sin.to(input_dtype), cos.to(input_dtype), half)
        assert not torch.equal(expected, early_cast)
    grad = torch.randn_like(expected)
    reference_grads = torch.autograd.grad(expected, (x, sin, cos), grad, retain_graph=True)
    actual_grads = torch.autograd.grad(apply_rot_embed_cat(x, cat_embed, half), (x, sin, cos), grad)
    _assert_all_equal(actual_grads, reference_grads)


def test_rotary_apply_list_preserves_each_dtype():
    sin, cos = build_rotary_pos_embed((3, 5), dim=32)
    inputs = [torch.randn(1, 15, 32, dtype=dtype) for dtype in (torch.bfloat16, torch.float32, torch.float64)]
    outputs = apply_rot_embed_list(inputs, sin, cos)
    assert [x.dtype for x in outputs] == [x.dtype for x in inputs]


def test_rope_frequency_buffers_with_low_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        for cls, name in (
                (RotaryEmbedding, 'bands'), (RotaryEmbeddingCat, 'bands'),
                (RotaryEmbeddingDinoV3, 'periods'), (RotaryEmbeddingMRope, 'inv_freq'),
        ):
            assert getattr(cls(64), name).dtype == torch.float32
        mixed = RotaryEmbeddingMixed(64, depth=2, num_heads=2, feat_shape=(3, 513))
        assert mixed.t_x.dtype == mixed.t_y.dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------------------------------------------------
# Coordinate augmentation: reference numerics, RNG consumption, caches, and model integration.
# ---------------------------------------------------------------------------------------------------------------------

_SHAPE = (5, 9)
_AUG = dict(shift_coords=0.3, jitter_coords=1.4, rescale_coords=1.8)
_KINDS = ['base', 'cat', 'cat_pixel', 'cat_centered', 'mixed', 'mrope', 'dinov3']
_CASES = [(kind, cached) for kind in _KINDS for cached in (False, True) if not (kind == 'mrope' and cached)]
_AUG_CASES = [dict(zip(_AUG, values)) for values in product(*[(None, v) for v in _AUG.values()])]
# Identity augmentation also exercises integer-valued configuration.
_AUG_CASES.append(dict(shift_coords=0, jitter_coords=1, rescale_coords=1))


def _legacy_augment(coords, shift_coords=None, jitter_coords=None, rescale_coords=None):
    # Preserve the pre-sharing DINOv3 implementation independently, including draw shape, device, dtype and order.
    device, dtype = coords.device, coords.dtype
    if shift_coords is not None:
        shift = float(shift_coords)
        shift_hw = torch.empty(2, device=device, dtype=dtype).uniform_(-shift, shift)
        coords = coords + shift_hw[None, :]
    if jitter_coords is not None:
        jitter_factor = float(jitter_coords)
        if jitter_factor <= 0:
            raise ValueError('jitter_coords must be > 0 (interpreted as multiplicative factor).')
        jitter_max = math.log(jitter_factor)
        jitter_hw = torch.empty(2, device=device, dtype=dtype).uniform_(-jitter_max, jitter_max).exp()
        coords = coords * jitter_hw[None, :]
    if rescale_coords is not None:
        rescale_factor = float(rescale_coords)
        if rescale_factor <= 0:
            raise ValueError('rescale_coords must be > 0 (interpreted as multiplicative factor).')
        rescale_max = math.log(rescale_factor)
        rescale = torch.empty(1, device=device, dtype=dtype).uniform_(-rescale_max, rescale_max).exp()
        coords = coords * rescale
    return coords


def _make(kind, cached=False, device='cpu', dtype=torch.float32, **aug):
    kwargs = dict(device=device, dtype=dtype, **aug)
    if kind == 'mrope':
        return RotaryEmbeddingMRope(64, **kwargs)
    kwargs['feat_shape'] = _SHAPE if cached else None
    if kind == 'mixed':
        return RotaryEmbeddingMixed(64, depth=2, num_heads=2, grid_indexing='xy', **kwargs)
    if kind == 'dinov3':
        return RotaryEmbeddingDinoV3(64, normalize_coords='max', grid_offset=0.25, **kwargs)
    if kind == 'cat_pixel':
        kwargs.update(in_pixels=True, linear_bands=True)
    elif kind == 'cat_centered':
        kwargs.update(grid_type='centered', normalize_coords='min', in_pixels=False)
    else:
        kwargs.update(in_pixels=False, ref_feat_shape=(7, 11), grid_offset=0.5, grid_indexing='xy')
    return (RotaryEmbedding if kind == 'base' else RotaryEmbeddingCat)(64, **kwargs)


def _cat(embed):
    return torch.cat(embed, -1) if isinstance(embed, tuple) else embed


def _rng(device):
    states = [torch.get_rng_state()]
    if device.type == 'cuda':
        states.append(torch.cuda.get_rng_state(device))
    return states


def _reference_embed(m, aug):
    output_device = None
    if isinstance(m, RotaryEmbeddingDinoV3):
        # Grid and augmentation remain on CPU, even when periods/model live on CUDA.
        coords = _build_grid(
            _SHAPE, grid_type='centered', normalize_coords=m.normalize_coords,
            grid_offset=m.grid_offset, grid_indexing=m.grid_indexing,
        ).reshape(-1, 2)
        coords = _legacy_augment(coords, **aug).to(device=m.periods.device, dtype=m.periods.dtype)
        angles = (2 * math.pi * coords[:, :, None] / m.periods[None, None, :]).flatten(1)
        angles = angles.tile(2) if m.rotate_half else angles.repeat_interleave(2, -1)
    elif isinstance(m, RotaryEmbeddingMixed):
        t_x, t_y = get_mixed_grid(list(_SHAPE), grid_indexing=m.grid_indexing, device=m.freqs.device)
        coords = _legacy_augment(torch.stack([t_x, t_y], -1), **aug)
        return _legacy_mixed_embed(m.freqs.float(), coords[:, 0], coords[:, 1])
    elif isinstance(m, RotaryEmbeddingMRope):
        coords = _build_grid(_SHAPE, device=m.inv_freq.device).reshape(-1, 2)
        coords = _legacy_augment(coords, **aug)
        pos = torch.zeros(coords.shape[0], m.dim // 2, device=m.inv_freq.device)
        pos[:, 1:m.mrope_section[1] * 3:3] = coords[:, :1]
        pos[:, 2:m.mrope_section[2] * 3:3] = coords[:, 1:]
        angles = (pos * m.inv_freq).tile(2)
    else:
        bands = m.bands
        if m._use_cached_embed:
            buf = m.pos_embed_sin if isinstance(m, RotaryEmbedding) else m.pos_embed
            output_device = buf.device
            # Unaugmented caches are computed on CPU at construction, then copied to the target device.
            bands = m._compute_bands(device=buf.device if any(v is not None for v in aug.values()) else None)
        coords = _build_grid(
            _SHAPE, grid_type=m.grid_type, ref_feat_shape=m.ref_feat_shape, grid_offset=m.grid_offset,
            grid_indexing=m.grid_indexing, normalize_coords=m.normalize_coords, device=bands.device,
        ).reshape(-1, 2)
        coords = _legacy_augment(coords, **aug)
        angles = (coords[:, :, None] * bands).flatten(1)
        angles = angles.tile(2) if getattr(m, 'rotate_half', False) else angles.repeat_interleave(2, -1)
    return torch.cat([angles.sin(), angles.cos()], -1).to(device=output_device, dtype=torch.float32)


@pytest.mark.parametrize('kind,cached', _CASES)
@pytest.mark.parametrize('aug', _AUG_CASES)
def test_rope_aug_reference_and_rng(kind, cached, aug, rope_device):
    m = _make(kind, cached, device=rope_device, **aug).train()
    buffers_before = [b.clone() for b in m.buffers()]
    torch.manual_seed(123)
    expected = _reference_embed(m, aug)
    rng_expected = _rng(rope_device)
    torch.manual_seed(123)
    actual = _cat(m.get_embed(None if cached else _SHAPE, dtype=torch.float32))
    _assert_all_equal(actual, expected)
    _assert_all_equal(_rng(rope_device), rng_expected)
    _assert_all_equal(list(m.buffers()), buffers_before)
    if kind == 'mixed' and any(v is not None for v in aug.values()):
        actual.mean().backward()
        assert m.freqs.grad is not None and torch.isfinite(m.freqs.grad).all()
        assert m.freqs.grad.abs().sum() > 0
    if kind == 'mrope':
        temporal = (m.axis == 0).repeat(2)
        sin, cos = actual.chunk(2, -1)
        assert not sin[:, temporal].any()
        assert (cos[:, temporal] == 1).all()


@pytest.mark.parametrize('kind,cached', _CASES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_rope_aug_eval_initialization_and_caches(kind, cached, dtype, rope_device):
    torch.manual_seed(321)
    plain = _make(kind, cached, device=rope_device, dtype=dtype)
    rng_plain = _rng(rope_device)
    torch.manual_seed(321)
    augmented = _make(kind, cached, device=rope_device, dtype=dtype, **_AUG)
    _assert_all_equal(_rng(rope_device), rng_plain)  # enabling augmentation must not sample in constructors
    assert dict(augmented.named_buffers()).keys() == dict(plain.named_buffers()).keys()
    for name, buffer in plain.named_buffers():
        _assert_all_equal(augmented.get_buffer(name), buffer)
    _assert_all_equal(list(augmented.parameters()), list(plain.parameters()))
    plain.eval()
    augmented.eval()
    for shape in ([None, _SHAPE, (7, 3)] if cached else [_SHAPE, (7, 3)]):
        rng_before = _rng(rope_device)
        _assert_all_equal(augmented.get_embed(shape), plain.get_embed(shape))
        _assert_all_equal(_rng(rope_device), rng_before)
    if kind != 'mrope':
        augmented.train()
        rng_before = _rng(rope_device)
        augmented.init_non_persistent_buffers()
        if cached:
            augmented.update_feat_shape((7, 3))
            plain.update_feat_shape((7, 3))
        _assert_all_equal(_rng(rope_device), rng_before)
        assert dict(augmented.named_buffers()).keys() == dict(plain.named_buffers()).keys()
        for name, buffer in plain.named_buffers():
            _assert_all_equal(augmented.get_buffer(name), buffer)
    augmented.train()
    shape = (7, 3) if cached else _SHAPE
    augmented.get_embed(shape)
    augmented.eval()
    _assert_all_equal(augmented.get_embed(shape), plain.get_embed(shape))


@pytest.mark.parametrize('kind', ['cat', 'cat_pixel', 'cat_centered', 'mixed', 'dinov3'])
@pytest.mark.parametrize('seq_len', [None, 60])
def test_rope_aug_batch_sampling(kind, seq_len, rope_device):
    m = _make(kind, device=rope_device, **_AUG).train()
    shapes = [(5, 9), (3, 7), (5, 9)]
    torch.manual_seed(234)
    expected = [m.get_embed(shape, dtype=torch.bfloat16) for shape in shapes]
    rng_expected = _rng(rope_device)
    torch.manual_seed(234)
    out = m.get_batch_embeds(shapes, seq_len=seq_len, dtype=torch.bfloat16)
    _assert_all_equal(_rng(rope_device), rng_expected)
    if seq_len is None:
        _assert_all_equal(out, expected)
    else:
        for i, embed in enumerate(expected):
            n = embed.shape[-2]
            _assert_all_equal(out[i, ..., :n, :], embed)
            assert not out[i, ..., n:, :].any()
    assert not torch.equal(expected[0], expected[2])  # independent draw for each shapes entry


@pytest.mark.parametrize('kind', _KINDS)
def test_rope_aug_dynamic_fx(kind):
    m = _GetEmbedWrapper(_make(kind, **_AUG)).train()
    traced = torch.fx.symbolic_trace(m)
    for seed, shape in ((12, (5, 9)), (34, (3, 7))):
        x = torch.zeros(1, 1, *shape)
        torch.manual_seed(seed)
        expected = m(x)
        rng_expected = torch.get_rng_state()
        torch.manual_seed(seed)
        _assert_all_equal(traced(x), expected)
        _assert_all_equal(torch.get_rng_state(), rng_expected)


@pytest.mark.parametrize('kind', _KINDS)
def test_rope_aug_autocast(kind, rope_device):
    m = _make(kind, device=rope_device, **_AUG)
    torch.manual_seed(456)
    expected = m.get_embed(_SHAPE)
    rng_expected = _rng(rope_device)
    dtypes = [torch.bfloat16, torch.float16] if rope_device.type == 'cuda' else [torch.bfloat16]
    for dtype in dtypes:
        torch.manual_seed(456)
        with torch.autocast(rope_device.type, dtype=dtype):
            _assert_all_equal(m.get_embed(_SHAPE), expected)
        _assert_all_equal(_rng(rope_device), rng_expected)


@pytest.mark.parametrize('rope_type', ['axial', 'mixed', 'mrope', 'dinov3'])
@pytest.mark.parametrize('dict_input', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_naflex_rope_aug_configuration(rope_type, dict_input, dtype, rope_device):
    from timm import create_model
    kwargs = dict(embed_dim=64, depth=2, num_heads=2, num_classes=5, rope_type=rope_type)
    if rope_type == 'mrope':
        kwargs['rope_mrope_section'] = (4, 6, 6)
    model = create_model('naflexvit_base_patch16_gap', **kwargs, **{'rope_' + k: v for k, v in _AUG.items()})
    model.to(device=rope_device, dtype=dtype)
    assert model.rope.aug_active
    for k, v in _AUG.items():
        assert getattr(model.rope, k) == v
    x = torch.randn(2, 3, 80, 144, device=rope_device, dtype=dtype)
    if dict_input:
        coord = torch.zeros(2, 45, 2, dtype=torch.long, device=rope_device)
        coord[0, :, 0] = torch.arange(45, device=rope_device) // 9
        coord[0, :, 1] = torch.arange(45, device=rope_device) % 9
        coord[1, :21, 0] = torch.arange(21, device=rope_device) // 7
        coord[1, :21, 1] = torch.arange(21, device=rope_device) % 7
        valid = torch.zeros(2, 45, dtype=torch.bool, device=rope_device)
        valid[0] = True
        valid[1, :21] = True
        x = dict(
            patches=torch.randn(2, 45, 16 * 16 * 3, device=rope_device, dtype=dtype),
            patch_coord=coord, patch_valid=valid,
        )
    model.train()
    model(x).mean().backward()
    if rope_type == 'mixed':
        assert model.rope.freqs.grad is not None and torch.isfinite(model.rope.freqs.grad).all()
    model.eval()
    rng_before = torch.get_rng_state()
    with torch.no_grad():
        expected = model(x)
        model.rope.aug_active = False
        _assert_all_equal(model(x), expected)
    _assert_all_equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize('rope_type', ['cat', 'mixed', 'dinov3', 'mrope'])
def test_eva_rope_aug_configuration(rope_type):
    from timm.models.eva import Eva
    model = Eva(
        img_size=(80, 144), patch_size=16, embed_dim=64, depth=2, num_heads=2, num_classes=5,
        rope_type=rope_type, use_rot_pos_emb=True, **{'rope_' + k: v for k, v in _AUG.items()},
    )
    assert model.rope.aug_active
    for k, v in _AUG.items():
        assert getattr(model.rope, k) == v
    x = torch.randn(2, 3, 80, 144)
    model.train()
    model(x).mean().backward()
    model.eval()
    rng_before = torch.get_rng_state()
    with torch.no_grad():
        expected = model(x)
        model.rope.aug_active = False
        _assert_all_equal(model(x), expected)
    _assert_all_equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize('kind', _KINDS)
@pytest.mark.parametrize('name,value', [
    ('shift_coords', -0.1), ('shift_coords', float('inf')), ('shift_coords', float('nan')),
    ('jitter_coords', 0.5), ('jitter_coords', 0.), ('jitter_coords', -1.),
    ('jitter_coords', float('inf')), ('jitter_coords', float('nan')),
    ('rescale_coords', 0.5), ('rescale_coords', 0.), ('rescale_coords', -1.),
    ('rescale_coords', float('inf')), ('rescale_coords', float('nan')),
])
def test_rope_aug_invalid_config(kind, name, value):
    rng_before = torch.get_rng_state()
    with pytest.raises(ValueError, match=name):
        _make(kind, **{name: value})
    _assert_all_equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize('name,value', [('shift_coords', -1.), ('jitter_coords', 0.5), ('rescale_coords', 0.5)])
def test_rope_aug_invalid_functional_config(name, value):
    rng_before = torch.get_rng_state()
    with pytest.raises(ValueError, match=name):
        build_rotary_pos_embed((3, 5), **{name: value})
    _assert_all_equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize('kind,cached', _CASES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float64])
def test_rope_aug_output_dtype(kind, cached, dtype, rope_device):
    m = _make(kind, cached, device=rope_device, **_AUG).to(dtype=dtype)
    shape = None if cached else _SHAPE
    m.eval()
    eval_before = _cat(m.get_embed(shape)).clone()
    m.train()
    torch.manual_seed(321)
    generated = m.get_embed(shape)
    assert _cat(generated).dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    for requested_dtype in (torch.float32, dtype):
        torch.manual_seed(321)
        expected = _cat(generated).to(requested_dtype)
        _assert_all_equal(_cat(m.get_embed(shape, dtype=requested_dtype)), expected)
    m.eval()
    _assert_all_equal(_cat(m.get_embed(shape)), eval_before)


@pytest.mark.parametrize('kind', ['base', 'cat'])
def test_rope_cached_aug_reuses_bands(kind, rope_device, monkeypatch):
    m = _make(kind, cached=True, device=rope_device, **_AUG).bfloat16()
    assert m.bands.dtype == torch.float32 and m.bands.device.type == rope_device.type
    _assert_all_equal(m.bands, m._compute_bands(device=rope_device))
    assert 'bands' not in m.state_dict()
    bands = m.bands

    def fail_rebuild(*args, **kwargs):
        raise AssertionError('cached augmentation must not rebuild or copy frequency bands')

    monkeypatch.setattr(m, '_compute_bands', fail_rebuild)
    for _ in range(2):
        embed = _cat(m.get_embed())
        assert embed.device == bands.device and embed.dtype == torch.float32
        assert m.bands is bands


def _assert_independent_grid(t_x, t_y):
    assert t_x.is_contiguous() and t_y.is_contiguous()
    assert t_x._base is None and t_y._base is None
    assert t_x.untyped_storage().data_ptr() != t_y.untyped_storage().data_ptr()


@pytest.mark.parametrize('shape', [(1, 1), (5, 9)])
@pytest.mark.parametrize('grid_indexing', ['ij', 'xy'])
def test_mixed_grid_storage(shape, grid_indexing, rope_device):
    t_x, t_y = get_mixed_grid(shape, grid_indexing=grid_indexing, device=rope_device)
    _assert_independent_grid(t_x, t_y)
    ys, xs = torch.meshgrid(
        torch.arange(shape[0], device=rope_device), torch.arange(shape[1], device=rope_device), indexing='ij',
    )
    expected = [ys.flatten().float(), xs.flatten().float()]
    if grid_indexing == 'xy':
        expected.reverse()
    _assert_all_equal([t_x, t_y], expected)
    t_x.add_(123)
    _assert_all_equal(t_y, expected[1])
    m = RotaryEmbeddingMixed(64, 2, 2, feat_shape=(2, 3), grid_indexing=grid_indexing, device=rope_device)
    _assert_independent_grid(m.t_x, m.t_y)
    m.update_feat_shape(shape)
    for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float32):
        m.to(dtype=dtype)
        _assert_independent_grid(m.t_x, m.t_y)
        _assert_all_equal([m.t_x, m.t_y], expected)


@pytest.mark.parametrize('shape', [(), (3,), (2, 3, 4)])
def test_mixed_grid_rejects_non_2d(shape):
    with pytest.raises(ValueError, match='Expected 2 spatial dimensions'):
        get_mixed_grid(shape)
    rng_before = torch.get_rng_state()
    with pytest.raises(ValueError, match='Expected 2 spatial dimensions'):
        RotaryEmbeddingMixed(64, 2, 2, feat_shape=shape)
    _assert_all_equal(torch.get_rng_state(), rng_before)
    with pytest.raises(ValueError, match='Expected 2 spatial dimensions'):
        RotaryEmbeddingDinoV3(64, feat_shape=shape)


class _SpatialShapeEmbedWrapper(_GetEmbedWrapper):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rope.get_embed(shape=x.shape[2:])


@pytest.mark.parametrize('kind', ['mixed', 'mrope', 'dinov3'])
@pytest.mark.parametrize('shape', [(), (3,), (2, 3, 4)])
def test_rope_rejects_non_2d(kind, shape):
    m = _SpatialShapeEmbedWrapper(_make(kind))
    x = torch.zeros(1, 1, *shape)
    for module in (m, torch.fx.symbolic_trace(m)):
        with pytest.raises(ValueError, match='Expected 2 spatial dimensions'):
            module(x)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize('grid_indexing', ['ij', 'xy'])
def test_dinov3_coords_rounding(dtype, grid_indexing, rope_device):
    # The helper historically casts normalized coordinates before the final scale/shift.
    h = ((torch.arange(0.5, 5, device=rope_device) + 0.25) / 9).to(dtype)
    w = ((torch.arange(0.5, 9, device=rope_device) + 0.25) / 9).to(dtype)
    expected = 2. * torch.stack(torch.meshgrid(h, w, indexing='ij'), -1).flatten(0, 1) - 1.
    if grid_indexing == 'xy':
        expected = expected.flip(-1)
    actual = make_coords_dinov3(
        5, 9, normalize_coords='max', grid_indexing=grid_indexing, grid_offset=0.25, device=rope_device, dtype=dtype,
    )
    _assert_all_equal(actual, expected)


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('sections', [None, (4, 2, 3)])
def test_eva_mrope_resize(dynamic, sections):
    from timm.models.eva import Eva
    model = Eva(
        img_size=(80, 144), patch_size=16, embed_dim=64, depth=2, num_heads=2, num_classes=5,
        rope_type='mrope', use_rot_pos_emb=True, dynamic_img_size=dynamic, rope_mrope_section=sections,
    ).eval()
    assert tuple(model.rope.mrope_section) == (sections or (8, 12, 12))
    assert all(blk.attn.rotate_half for blk in model.blocks)
    with torch.no_grad():
        assert model(torch.randn(1, 3, 80, 144)).shape == (1, 5)
        if not dynamic:
            model.set_input_size(img_size=(112, 64))
        x = torch.randn(1, 3, 112, 64)
        assert model(x).shape == (1, 5)
        # The rope table fed to the blocks must be built for the resized (7, 4) grid, not the original one.
        _, rope = model._pos_embed(model.patch_embed(x))
        expected = model.rope.get_embed((7, 4))
        assert rope.shape[-2] == 28
        _assert_all_equal(rope.reshape(expected.shape), expected)


@pytest.mark.parametrize('kind', _KINDS)
@pytest.mark.parametrize('name', list(_AUG))
@pytest.mark.parametrize('value', [False, True, '1.5'])
def test_rope_aug_rejects_non_numeric_config(kind, name, value):
    rng_before = torch.get_rng_state()
    with pytest.raises(TypeError, match=name):
        _make(kind, **{name: value})
    with pytest.raises(TypeError, match=name):
        build_rotary_pos_embed((3, 5), **{name: value})
    _assert_all_equal(torch.get_rng_state(), rng_before)


def test_rope_aug_accepts_tensor_scalars():
    values = {name: torch.tensor(value) for name, value in _AUG.items()}
    m = RotaryEmbeddingCat(64, **values)
    assert m.aug_active
    for name, value in values.items():
        assert getattr(m, name) == float(value)
    for bad in (torch.tensor([0.1, 0.2]), torch.tensor(True)):
        with pytest.raises(TypeError, match='shift_coords'):
            RotaryEmbeddingCat(64, shift_coords=bad)


@pytest.mark.parametrize('cls', [RotaryEmbedding, RotaryEmbeddingCat])
@pytest.mark.parametrize('in_pixels', [False, True])
def test_rope_cached_aug_enable_and_refresh(cls, in_pixels, rope_device):
    kwargs = dict(in_pixels=in_pixels, temperature=17., max_res=80, linear_bands=True, device=rope_device)
    m = cls(64, feat_shape=_SHAPE, **kwargs).bfloat16()
    m.eval()
    cached = _cat(m.get_embed()).clone()
    for name, value in _AUG.items():
        setattr(m, name, value)
    m.aug_active = True
    m.train()
    reference = cls(64, feat_shape=_SHAPE, **kwargs, **_AUG).bfloat16().train()
    torch.manual_seed(123)
    expected = reference.get_embed()
    rng_expected = _rng(rope_device)
    torch.manual_seed(123)
    _assert_all_equal(m.get_embed(), expected)
    _assert_all_equal(_rng(rope_device), rng_expected)
    m.eval()
    _assert_all_equal(_cat(m.get_embed()), cached)

    # Resizing must refresh both the eval cache and the retained bands after a configuration change.
    m.temperature, m.max_res, m.linear_bands = 31., 64, False
    m.update_feat_shape((7, 3))
    kwargs.update(temperature=31., max_res=64, linear_bands=False)
    reference = cls(64, **kwargs, **_AUG).bfloat16()
    _assert_all_equal(m.bands, reference.bands)
    reference.eval()
    _assert_all_equal(m.get_embed(), reference.get_embed((7, 3), dtype=torch.bfloat16))
    m.train()
    reference.train()
    torch.manual_seed(456)
    expected = reference.get_embed((7, 3))
    torch.manual_seed(456)
    _assert_all_equal(m.get_embed(), expected)


@pytest.mark.parametrize('family', ['eva', 'naflex'])
@pytest.mark.parametrize('rope_type', ['cat', 'dinov3'])
@pytest.mark.parametrize('half', [False, True])
def test_model_rope_layout(family, rope_type, half):
    from timm import create_model
    from timm.models.eva import Eva
    kwargs = dict(embed_dim=64, depth=2, num_heads=2, num_classes=5, rope_rotate_half=half)
    if family == 'eva':
        model = Eva(img_size=(80, 144), patch_size=16, use_rot_pos_emb=True, rope_type=rope_type, **kwargs)
    else:
        model = create_model(
            'naflexvit_base_patch16_gap', rope_type='axial' if rope_type == 'cat' else rope_type, **kwargs,
        )
    model.eval()
    assert model.rope.rotate_half == half
    assert all(blk.attn.rotate_half == half for blk in model.blocks)
    # The requested layout must pair each sin/cos channel with the corresponding rotation channel.
    if rope_type == 'cat':
        reference = build_rotary_pos_embed((5, 9), dim=32, in_pixels=False, rotate_half=half)
        reference = torch.cat(reference, -1)
    else:
        coords = make_coords_dinov3(5, 9)
        angles = (2 * math.pi * coords[:, :, None] / model.rope.periods).flatten(1)
        angles = angles.tile(2) if half else angles.repeat_interleave(2, -1)
        reference = torch.cat([angles.sin(), angles.cos()], -1)
    _assert_all_equal(model.rope.get_embed((5, 9)), reference)


@pytest.mark.parametrize('rope_type,half', [('mixed', True), ('mrope', False)])
def test_rope_factory_rejects_unsupported_layout(rope_type, half):
    with pytest.raises(ValueError, match='requires rotate_half='):
        create_rope_embed(rope_type, dim=64, num_heads=2, rotate_half=half)


@pytest.mark.parametrize('family', ['eva', 'naflex'])
def test_model_mixed_rejects_half_layout(family):
    from timm import create_model
    from timm.models.eva import Eva
    kwargs = dict(embed_dim=64, depth=2, num_heads=2, rope_type='mixed', rope_rotate_half=True)
    with pytest.raises(ValueError, match='requires .*rotate_half=False'):
        if family == 'eva':
            Eva(img_size=32, use_rot_pos_emb=True, **kwargs)
        else:
            create_model('naflexvit_base_patch16_gap', **kwargs)


@pytest.mark.parametrize('qkv_separate', [False, True])
def test_mrope_attention_pool_layout(qkv_separate):
    from timm.layers import RotAttentionPool2d
    pool = RotAttentionPool2d(64, num_heads=2, rope_type='mrope', qkv_separate=qkv_separate).eval()
    x = torch.randn(2, 64, 3, 5)
    tokens = x.flatten(2).transpose(1, 2)
    tokens = torch.cat([tokens.mean(1, keepdim=True), tokens], 1)
    if qkv_separate:
        q, k, v = [proj(tokens).reshape(2, 16, 2, 32).transpose(1, 2) for proj in (pool.q, pool.k, pool.v)]
    else:
        q, k, v = pool.qkv(tokens).reshape(2, 16, 3, 2, 32).permute(2, 0, 3, 1, 4).unbind(0)
    rope = pool.pos_embed.get_embed((3, 5))
    q = torch.cat([q[:, :, :1], apply_rot_embed_cat(q[:, :, 1:], rope, half=True)], 2)
    k = torch.cat([k[:, :, :1], apply_rot_embed_cat(k[:, :, 1:], rope, half=True)], 2)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 16, 64)
    expected = pool.proj(expected)[:, 0]
    with torch.no_grad():
        torch.testing.assert_close(pool(x), expected)


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('rope_type', [None, 'cat', 'mixed', 'dinov3', 'mrope'])
def test_eva_rope_fx(rope_type, dynamic):
    from timm.layers import get_notrace_functions, get_notrace_modules
    from timm.models.eva import Eva
    model = Eva(
        img_size=(80, 144), patch_size=16, embed_dim=64, depth=2, num_heads=2, num_classes=5,
        use_rot_pos_emb=rope_type is not None, rope_type=rope_type, dynamic_img_size=dynamic,
    ).eval()
    tracer = torch.fx.Tracer(autowrap_functions=get_notrace_functions())
    leaf_types = tuple(get_notrace_modules())
    default_is_leaf = tracer.is_leaf_module
    tracer.is_leaf_module = lambda module, name: isinstance(module, leaf_types) or default_is_leaf(module, name)
    traced = torch.fx.GraphModule(model, tracer.trace(model))
    for shape in ([(80, 144), (112, 64)] if dynamic else [(80, 144)]):
        x = torch.randn(1, 3, *shape)
        with torch.no_grad():
            expected = model(x)
            _assert_all_equal(traced(x), expected)
