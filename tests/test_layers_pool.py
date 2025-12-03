"""Tests for timm pooling layers."""
import pytest
import torch
import torch.nn as nn

import importlib
import os

torch_backend = os.environ.get('TORCH_BACKEND')
if torch_backend is not None:
    importlib.import_module(torch_backend)
torch_device = os.environ.get('TORCH_DEVICE', 'cpu')


# Adaptive Avg/Max Pooling Tests

class TestAdaptiveAvgMaxPool:
    """Test adaptive_avgmax_pool module."""

    def test_adaptive_avgmax_pool2d(self):
        from timm.layers import adaptive_avgmax_pool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        out = adaptive_avgmax_pool2d(x, 1)
        assert out.shape == (2, 64, 1, 1)
        # Should be average of avg and max
        expected = 0.5 * (x.mean(dim=(2, 3), keepdim=True) + x.amax(dim=(2, 3), keepdim=True))
        assert torch.allclose(out, expected)

    def test_select_adaptive_pool2d(self):
        from timm.layers import select_adaptive_pool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)

        out_avg = select_adaptive_pool2d(x, pool_type='avg', output_size=1)
        assert out_avg.shape == (2, 64, 1, 1)
        assert torch.allclose(out_avg, x.mean(dim=(2, 3), keepdim=True))

        out_max = select_adaptive_pool2d(x, pool_type='max', output_size=1)
        assert out_max.shape == (2, 64, 1, 1)
        assert torch.allclose(out_max, x.amax(dim=(2, 3), keepdim=True))

    def test_adaptive_avgmax_pool2d_module(self):
        from timm.layers import AdaptiveAvgMaxPool2d
        x = torch.randn(2, 64, 14, 14, device=torch_device)
        pool = AdaptiveAvgMaxPool2d(output_size=1).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64, 1, 1)

    def test_select_adaptive_pool2d_module(self):
        from timm.layers import SelectAdaptivePool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)

        for pool_type in ['avg', 'max', 'avgmax', 'catavgmax']:
            pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True).to(torch_device)
            out = pool(x)
            if pool_type == 'catavgmax':
                assert out.shape == (2, 128)  # concatenated
            else:
                assert out.shape == (2, 64)

    def test_select_adaptive_pool2d_fast(self):
        from timm.layers import SelectAdaptivePool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)

        for pool_type in ['fast', 'fastavg', 'fastmax', 'fastavgmax', 'fastcatavgmax']:
            pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True).to(torch_device)
            out = pool(x)
            if 'cat' in pool_type:
                assert out.shape == (2, 128)
            else:
                assert out.shape == (2, 64)


# Attention Pool Tests

class TestAttentionPool:
    """Test attention-based pooling layers."""

    def test_attention_pool_latent_basic(self):
        from timm.layers import AttentionPoolLatent
        x = torch.randn(2, 49, 64, device=torch_device)
        pool = AttentionPoolLatent(in_features=64, num_heads=4).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_attention_pool_latent_multi_latent(self):
        from timm.layers import AttentionPoolLatent
        x = torch.randn(2, 49, 64, device=torch_device)
        pool = AttentionPoolLatent(
            in_features=64,
            num_heads=4,
            latent_len=4,
            pool_type='avg',
        ).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_attention_pool2d_basic(self):
        from timm.layers import AttentionPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = AttentionPool2d(in_features=64, feat_size=7).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_attention_pool2d_different_feat_size(self):
        from timm.layers import AttentionPool2d
        # Test with different spatial sizes (requires pos_embed interpolation)
        pool = AttentionPool2d(in_features=64, feat_size=7).to(torch_device)
        for size in [7, 14]:
            x = torch.randn(2, 64, size, size, device=torch_device)
            out = pool(x)
            assert out.shape == (2, 64)

    def test_rot_attention_pool2d_basic(self):
        from timm.layers import RotAttentionPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = RotAttentionPool2d(in_features=64, ref_feat_size=7).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_rot_attention_pool2d_different_sizes(self):
        from timm.layers import RotAttentionPool2d
        pool = RotAttentionPool2d(in_features=64, ref_feat_size=7).to(torch_device)
        for size in [7, 14, 10]:
            x = torch.randn(2, 64, size, size, device=torch_device)
            out = pool(x)
            assert out.shape == (2, 64)

    def test_rot_attention_pool2d_rope_types(self):
        from timm.layers import RotAttentionPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        for rope_type in ['base', 'cat', 'dinov3']:
            pool = RotAttentionPool2d(
                in_features=64,
                ref_feat_size=7,
                rope_type=rope_type,
            ).to(torch_device)
            out = pool(x)
            assert out.shape == (2, 64)


# LSE Pool Tests

class TestLsePool:
    """Test LogSumExp pooling layers."""

    def test_lse_plus_2d_basic(self):
        from timm.layers import LsePlus2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = LsePlus2d().to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64, 1, 1)

    def test_lse_plus_2d_flatten(self):
        from timm.layers import LsePlus2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = LsePlus2d(flatten=True).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_lse_plus_1d_basic(self):
        from timm.layers import LsePlus1d
        x = torch.randn(2, 49, 64, device=torch_device)
        pool = LsePlus1d().to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_lse_high_r_approximates_max(self):
        from timm.layers import LsePlus2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = LsePlus2d(r=100.0, r_learnable=False).to(torch_device)
        out = pool(x)
        out_max = x.amax(dim=(2, 3), keepdim=True)
        assert torch.allclose(out, out_max, atol=0.1)

    def test_lse_low_r_approximates_avg(self):
        from timm.layers import LsePlus2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = LsePlus2d(r=0.01, r_learnable=False).to(torch_device)
        out = pool(x)
        out_avg = x.mean(dim=(2, 3), keepdim=True)
        assert torch.allclose(out, out_avg, atol=0.1)

    def test_lse_learnable_r_gradient(self):
        from timm.layers import LsePlus2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = LsePlus2d(r=10.0, r_learnable=True).to(torch_device)
        out = pool(x).sum()
        out.backward()
        assert pool.r.grad is not None
        assert pool.r.grad.abs() > 0


# SimPool Tests

class TestSimPool:
    """Test SimPool attention-based pooling layers."""

    def test_simpool_2d_basic(self):
        from timm.layers import SimPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = SimPool2d(dim=64).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 1, 64)

    def test_simpool_2d_flatten(self):
        from timm.layers import SimPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = SimPool2d(dim=64, flatten=True).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_simpool_1d_basic(self):
        from timm.layers import SimPool1d
        x = torch.randn(2, 49, 64, device=torch_device)
        pool = SimPool1d(dim=64).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_simpool_multi_head(self):
        from timm.layers import SimPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        for num_heads in [1, 2, 4, 8]:
            pool = SimPool2d(dim=64, num_heads=num_heads, flatten=True).to(torch_device)
            out = pool(x)
            assert out.shape == (2, 64)

    def test_simpool_with_gamma(self):
        from timm.layers import SimPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = SimPool2d(dim=64, gamma=2.0, flatten=True).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()

    def test_simpool_qk_norm(self):
        from timm.layers import SimPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = SimPool2d(dim=64, qk_norm=True, flatten=True).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)


# Slot Pool Tests

class TestSlotPool:
    """Test Slot Attention pooling layers."""

    def test_slot_pool_basic(self):
        from timm.layers import SlotPool
        x = torch.randn(2, 49, 64, device=torch_device)
        pool = SlotPool(dim=64).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_slot_pool_2d_basic(self):
        from timm.layers import SlotPool2d
        x = torch.randn(2, 64, 7, 7, device=torch_device)
        pool = SlotPool2d(dim=64).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_slot_pool_multi_slot(self):
        from timm.layers import SlotPool
        x = torch.randn(2, 49, 64, device=torch_device)
        for num_slots in [1, 2, 4, 8]:
            pool = SlotPool(dim=64, num_slots=num_slots).to(torch_device)
            out = pool(x)
            assert out.shape == (2, 64)

    def test_slot_pool_iterations(self):
        from timm.layers import SlotPool
        x = torch.randn(2, 49, 64, device=torch_device)
        for iters in [1, 2, 3, 5]:
            pool = SlotPool(dim=64, iters=iters).to(torch_device)
            out = pool(x)
            assert out.shape == (2, 64)

    def test_slot_pool_pool_types(self):
        from timm.layers import SlotPool
        x = torch.randn(2, 49, 64, device=torch_device)
        for pool_type in ['max', 'avg', 'first']:
            pool = SlotPool(dim=64, num_slots=4, pool_type=pool_type).to(torch_device)
            out = pool(x)
            assert out.shape == (2, 64)

    def test_slot_pool_stochastic_train_mode(self):
        from timm.layers import SlotPool
        x = torch.randn(2, 49, 64, device=torch_device)
        pool = SlotPool(dim=64, stochastic_init=True).to(torch_device)
        pool.train()
        out1 = pool(x)
        out2 = pool(x)
        # Should differ in train mode with stochastic init
        assert not torch.allclose(out1, out2)

    def test_slot_pool_stochastic_eval_mode(self):
        from timm.layers import SlotPool
        x = torch.randn(2, 49, 64, device=torch_device)
        pool = SlotPool(dim=64, stochastic_init=True).to(torch_device)
        pool.eval()
        out1 = pool(x)
        out2 = pool(x)
        # Should be deterministic in eval mode
        assert torch.allclose(out1, out2)


# Common Tests (Gradient, JIT, dtype)

class TestPoolingCommon:
    """Common tests across all pooling layers."""

    @pytest.mark.parametrize('pool_cls,kwargs,input_shape', [
        ('LsePlus2d', {}, (2, 64, 7, 7)),
        ('LsePlus1d', {}, (2, 49, 64)),
        ('SimPool2d', {'dim': 64}, (2, 64, 7, 7)),
        ('SimPool1d', {'dim': 64}, (2, 49, 64)),
        ('SlotPool', {'dim': 64}, (2, 49, 64)),
        ('SlotPool2d', {'dim': 64}, (2, 64, 7, 7)),
        ('SelectAdaptivePool2d', {'pool_type': 'avg', 'flatten': True}, (2, 64, 7, 7)),
        ('AttentionPoolLatent', {'in_features': 64, 'num_heads': 4}, (2, 49, 64)),
        ('AttentionPool2d', {'in_features': 64, 'feat_size': 7}, (2, 64, 7, 7)),
        ('RotAttentionPool2d', {'in_features': 64, 'ref_feat_size': 7}, (2, 64, 7, 7)),
    ])
    def test_gradient_flow(self, pool_cls, kwargs, input_shape):
        import timm.layers as layers
        x = torch.randn(*input_shape, device=torch_device, requires_grad=True)
        pool = getattr(layers, pool_cls)(**kwargs).to(torch_device)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    @pytest.mark.parametrize('pool_cls,kwargs,input_shape', [
        ('LsePlus2d', {}, (2, 64, 7, 7)),
        ('LsePlus1d', {}, (2, 49, 64)),
        ('SimPool2d', {'dim': 64}, (2, 64, 7, 7)),
        ('SimPool1d', {'dim': 64}, (2, 49, 64)),
        ('SlotPool', {'dim': 64, 'iters': 2}, (2, 49, 64)),
        ('SlotPool2d', {'dim': 64, 'iters': 2}, (2, 64, 7, 7)),
        ('AttentionPool2d', {'in_features': 64, 'feat_size': 7}, (2, 64, 7, 7)),
        ('RotAttentionPool2d', {'in_features': 64, 'ref_feat_size': 7}, (2, 64, 7, 7)),
    ])
    def test_torchscript(self, pool_cls, kwargs, input_shape):
        import timm.layers as layers
        x = torch.randn(*input_shape, device=torch_device)
        pool = getattr(layers, pool_cls)(**kwargs).to(torch_device)
        pool.eval()
        scripted = torch.jit.script(pool)
        out_orig = pool(x)
        out_script = scripted(x)
        assert torch.allclose(out_orig, out_script, atol=1e-5)

    @pytest.mark.parametrize('pool_cls,kwargs,input_shape', [
        ('LsePlus2d', {'flatten': True}, (2, 64, 7, 7)),
        ('LsePlus1d', {}, (2, 49, 64)),
        ('SimPool2d', {'dim': 64, 'flatten': True}, (2, 64, 7, 7)),
        ('SimPool1d', {'dim': 64}, (2, 49, 64)),
        ('SlotPool', {'dim': 64}, (2, 49, 64)),
        ('SlotPool2d', {'dim': 64}, (2, 64, 7, 7)),
        ('AttentionPool2d', {'in_features': 64, 'feat_size': 7}, (2, 64, 7, 7)),
        ('RotAttentionPool2d', {'in_features': 64, 'ref_feat_size': 7}, (2, 64, 7, 7)),
    ])
    def test_eval_deterministic(self, pool_cls, kwargs, input_shape):
        import timm.layers as layers
        x = torch.randn(*input_shape, device=torch_device)
        pool = getattr(layers, pool_cls)(**kwargs).to(torch_device)
        pool.eval()
        with torch.no_grad():
            out1 = pool(x)
            out2 = pool(x)
        assert torch.allclose(out1, out2)

    @pytest.mark.parametrize('pool_cls,kwargs,input_shape', [
        ('LsePlus2d', {'flatten': True}, (2, 64, 7, 7)),
        ('SimPool2d', {'dim': 64, 'flatten': True}, (2, 64, 7, 7)),
        ('SlotPool2d', {'dim': 64}, (2, 64, 7, 7)),
        ('RotAttentionPool2d', {'in_features': 64, 'ref_feat_size': 7}, (2, 64, 7, 7)),
    ])
    def test_different_spatial_sizes(self, pool_cls, kwargs, input_shape):
        import timm.layers as layers
        B, C, _, _ = input_shape
        pool = getattr(layers, pool_cls)(**kwargs).to(torch_device)
        for H, W in [(7, 7), (14, 14), (1, 1), (3, 5)]:
            x = torch.randn(B, C, H, W, device=torch_device)
            out = pool(x)
            assert out.shape[0] == B
            assert out.shape[-1] == C


# BlurPool Tests

class TestBlurPool:
    """Test BlurPool anti-aliasing layer."""

    def test_blur_pool_2d_basic(self):
        from timm.layers import BlurPool2d
        x = torch.randn(2, 64, 14, 14, device=torch_device)
        pool = BlurPool2d(channels=64).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64, 7, 7)

    def test_blur_pool_2d_stride(self):
        from timm.layers import BlurPool2d
        x = torch.randn(2, 64, 28, 28, device=torch_device)
        pool = BlurPool2d(channels=64, stride=4).to(torch_device)
        out = pool(x)
        assert out.shape == (2, 64, 8, 8)


# Pool1d Tests

class TestPool1d:
    """Test 1D pooling utilities."""

    def test_global_pool_nlc(self):
        from timm.layers import global_pool_nlc
        x = torch.randn(2, 49, 64, device=torch_device)

        # By default, avg/max excludes first token (num_prefix_tokens=1)
        out_avg = global_pool_nlc(x, pool_type='avg')
        assert out_avg.shape == (2, 64)
        assert torch.allclose(out_avg, x[:, 1:].mean(dim=1))

        out_max = global_pool_nlc(x, pool_type='max')
        assert out_max.shape == (2, 64)
        assert torch.allclose(out_max, x[:, 1:].amax(dim=1))

        out_first = global_pool_nlc(x, pool_type='token')
        assert out_first.shape == (2, 64)
        assert torch.allclose(out_first, x[:, 0])

        # Test with reduce_include_prefix=True
        out_avg_all = global_pool_nlc(x, pool_type='avg', reduce_include_prefix=True)
        assert torch.allclose(out_avg_all, x.mean(dim=1))
