"""Tests for timm.layers.drop module (DropBlock, DropPath)."""
import torch
import pytest

from timm.layers.drop import drop_block_2d, DropBlock2d, drop_path, DropPath


class TestDropBlock2d:
    """Test drop_block_2d function and DropBlock2d module."""

    def test_drop_block_2d_output_shape(self):
        """Test that output shape matches input shape."""
        for h, w in [(7, 7), (4, 8), (10, 5), (3, 3)]:
            x = torch.ones((2, 3, h, w))
            result = drop_block_2d(x, drop_prob=0.1, block_size=3)
            assert result.shape == x.shape, f"Shape mismatch for input ({h}, {w})"

    def test_drop_block_2d_no_drop_when_prob_zero(self):
        """Test that no dropping occurs when drop_prob=0."""
        x = torch.ones((2, 3, 8, 8))
        result = drop_block_2d(x, drop_prob=0.0, block_size=3)
        assert torch.allclose(result, x)

    def test_drop_block_2d_approximate_keep_ratio(self):
        """Test that the drop ratio is approximately correct."""
        torch.manual_seed(123)
        # Use large batch for statistical stability
        x = torch.ones((32, 16, 56, 56))
        drop_prob = 0.1

        # With scale_by_keep=False, kept values stay at 1.0 and dropped are 0.0
        # so we can directly measure the drop ratio
        result = drop_block_2d(x, drop_prob=drop_prob, block_size=7, scale_by_keep=False)

        total_elements = result.numel()
        dropped_elements = (result == 0).sum().item()
        actual_drop_ratio = dropped_elements / total_elements

        # Allow some tolerance since it's stochastic
        assert abs(actual_drop_ratio - drop_prob) < 0.03, \
            f"Drop ratio {actual_drop_ratio:.3f} not close to expected {drop_prob}"

    def test_drop_block_2d_inplace(self):
        """Test inplace operation."""
        x = torch.ones((2, 3, 8, 8))
        x_clone = x.clone()
        torch.manual_seed(42)
        result = drop_block_2d(x_clone, drop_prob=0.3, block_size=3, inplace=True)
        assert result is x_clone, "Inplace should return the same tensor"

    def test_drop_block_2d_couple_channels_true(self):
        """Test couple_channels=True uses same mask for all channels."""
        torch.manual_seed(42)
        x = torch.ones((2, 4, 16, 16))
        result = drop_block_2d(x, drop_prob=0.3, block_size=5, couple_channels=True)

        # With couple_channels=True, all channels should have same drop pattern
        for b in range(x.shape[0]):
            mask_c0 = (result[b, 0] == 0).float()
            for c in range(1, x.shape[1]):
                mask_c = (result[b, c] == 0).float()
                assert torch.allclose(mask_c0, mask_c), f"Channel {c} has different mask than channel 0"

    def test_drop_block_2d_couple_channels_false(self):
        """Test couple_channels=False uses independent mask per channel."""
        torch.manual_seed(42)
        x = torch.ones((2, 4, 16, 16))
        result = drop_block_2d(x, drop_prob=0.3, block_size=5, couple_channels=False)

        # With couple_channels=False, channels should have different patterns
        # (with high probability for reasonable drop_prob)
        mask_c0 = (result[0, 0] == 0).float()
        mask_c1 = (result[0, 1] == 0).float()
        # They might occasionally be the same by chance, but very unlikely
        assert not torch.allclose(mask_c0, mask_c1), "Channels should have independent masks"

    def test_drop_block_2d_with_noise(self):
        """Test with_noise option adds gaussian noise to dropped regions."""
        torch.manual_seed(42)
        x = torch.ones((2, 3, 16, 16))
        result = drop_block_2d(x, drop_prob=0.3, block_size=5, with_noise=True)

        # With noise, dropped regions should have non-zero values from gaussian noise
        # The result should contain values other than the scaled kept values
        unique_vals = torch.unique(result)
        assert len(unique_vals) > 2, "With noise should produce varied values"

    def test_drop_block_2d_even_block_size(self):
        """Test that even block sizes work correctly."""
        x = torch.ones((2, 3, 16, 16))
        for block_size in [2, 4, 6]:
            result = drop_block_2d(x, drop_prob=0.1, block_size=block_size)
            assert result.shape == x.shape, f"Shape mismatch for block_size={block_size}"

    def test_drop_block_2d_asymmetric_input(self):
        """Test with asymmetric H != W inputs."""
        for h, w in [(8, 16), (16, 8), (7, 14), (14, 7)]:
            x = torch.ones((2, 3, h, w))
            result = drop_block_2d(x, drop_prob=0.1, block_size=5)
            assert result.shape == x.shape, f"Shape mismatch for ({h}, {w})"

    def test_drop_block_2d_scale_by_keep(self):
        """Test scale_by_keep parameter."""
        torch.manual_seed(42)
        x = torch.ones((2, 3, 16, 16))

        # With scale_by_keep=True (default), kept values are scaled up
        result_scaled = drop_block_2d(x.clone(), drop_prob=0.3, block_size=5, scale_by_keep=True)
        kept_vals_scaled = result_scaled[result_scaled > 0]
        # Scaled values should be > 1.0 (scaled up to compensate for drops)
        assert kept_vals_scaled.min() > 1.0, "Scaled values should be > 1.0"

        # With scale_by_keep=False, kept values stay at original
        torch.manual_seed(42)
        result_unscaled = drop_block_2d(x.clone(), drop_prob=0.3, block_size=5, scale_by_keep=False)
        kept_vals_unscaled = result_unscaled[result_unscaled > 0]
        # Unscaled values should be exactly 1.0
        assert torch.allclose(kept_vals_unscaled, torch.ones_like(kept_vals_unscaled)), \
            "Unscaled values should be 1.0"


class TestDropBlock2dModule:
    """Test DropBlock2d nn.Module."""

    def test_deprecated_args_accepted(self):
        """Test that deprecated args (batchwise, fast) are silently accepted."""
        # These should not raise
        module1 = DropBlock2d(drop_prob=0.1, batchwise=True)
        module2 = DropBlock2d(drop_prob=0.1, fast=False)
        module3 = DropBlock2d(drop_prob=0.1, batchwise=False, fast=True)
        assert module1.drop_prob == 0.1
        assert module2.drop_prob == 0.1
        assert module3.drop_prob == 0.1

    def test_unknown_args_warned(self):
        """Test that unknown kwargs emit a warning."""
        with pytest.warns(UserWarning, match="unexpected keyword argument 'unknown_arg'"):
            DropBlock2d(drop_prob=0.1, unknown_arg=True)

    def test_training_mode(self):
        """Test that dropping only occurs in training mode."""
        module = DropBlock2d(drop_prob=0.5, block_size=3)
        x = torch.ones((2, 3, 8, 8))

        # In eval mode, should return input unchanged
        module.eval()
        result = module(x)
        assert torch.allclose(result, x), "Should not drop in eval mode"

        # In train mode, should modify input
        module.train()
        torch.manual_seed(42)
        result = module(x)
        assert not torch.allclose(result, x), "Should drop in train mode"

    def test_couple_channels_parameter(self):
        """Test couple_channels parameter is passed through."""
        x = torch.ones((2, 4, 16, 16))

        # couple_channels=True (default)
        module_coupled = DropBlock2d(drop_prob=0.3, block_size=5, couple_channels=True)
        module_coupled.train()
        torch.manual_seed(42)
        result_coupled = module_coupled(x)

        # All channels should have same pattern
        mask_c0 = (result_coupled[0, 0] == 0).float()
        mask_c1 = (result_coupled[0, 1] == 0).float()
        assert torch.allclose(mask_c0, mask_c1)

        # couple_channels=False
        module_uncoupled = DropBlock2d(drop_prob=0.3, block_size=5, couple_channels=False)
        module_uncoupled.train()
        torch.manual_seed(42)
        result_uncoupled = module_uncoupled(x)

        # Channels should have different patterns
        mask_c0 = (result_uncoupled[0, 0] == 0).float()
        mask_c1 = (result_uncoupled[0, 1] == 0).float()
        assert not torch.allclose(mask_c0, mask_c1)


class TestDropPath:
    """Test drop_path function and DropPath module."""

    def test_no_drop_when_prob_zero(self):
        """Test that no dropping occurs when drop_prob=0."""
        x = torch.ones((4, 8, 16, 16))
        result = drop_path(x, drop_prob=0.0, training=True)
        assert torch.allclose(result, x)

    def test_no_drop_when_not_training(self):
        """Test that no dropping occurs when not training."""
        x = torch.ones((4, 8, 16, 16))
        result = drop_path(x, drop_prob=0.5, training=False)
        assert torch.allclose(result, x)

    def test_drop_path_scaling(self):
        """Test that scale_by_keep properly scales kept paths."""
        torch.manual_seed(42)
        x = torch.ones((100, 8, 4, 4))  # Large batch for statistical stability
        keep_prob = 0.8
        drop_prob = 1 - keep_prob

        result = drop_path(x, drop_prob=drop_prob, training=True, scale_by_keep=True)

        # Kept samples should be scaled by 1/keep_prob = 1.25
        kept_mask = (result[:, 0, 0, 0] != 0)
        if kept_mask.any():
            kept_vals = result[kept_mask, 0, 0, 0]
            expected_scale = 1.0 / keep_prob
            assert torch.allclose(kept_vals, torch.full_like(kept_vals, expected_scale), atol=1e-5)

    def test_drop_path_no_scaling(self):
        """Test that scale_by_keep=False does not scale."""
        torch.manual_seed(42)
        x = torch.ones((100, 8, 4, 4))
        result = drop_path(x, drop_prob=0.2, training=True, scale_by_keep=False)

        # Kept samples should remain at 1.0
        kept_mask = (result[:, 0, 0, 0] != 0)
        if kept_mask.any():
            kept_vals = result[kept_mask, 0, 0, 0]
            assert torch.allclose(kept_vals, torch.ones_like(kept_vals))


class TestDropPathModule:
    """Test DropPath nn.Module."""

    def test_training_mode(self):
        """Test that dropping only occurs in training mode."""
        module = DropPath(drop_prob=0.5)
        x = torch.ones((32, 8, 4, 4))  # Larger batch for statistical reliability

        module.eval()
        result = module(x)
        assert torch.allclose(result, x), "Should not drop in eval mode"

        module.train()
        torch.manual_seed(42)
        result = module(x)
        # With 50% drop prob on 32 samples, very unlikely all survive
        # Check that at least one sample has zeros (was dropped)
        has_zeros = (result == 0).any()
        assert has_zeros, "Should drop some paths in train mode"

    def test_extra_repr(self):
        """Test extra_repr for nice printing."""
        module = DropPath(drop_prob=0.123)
        repr_str = module.extra_repr()
        assert "0.123" in repr_str
