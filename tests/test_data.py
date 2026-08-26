import numpy as np
import pytest
import torch

from timm.data import create_loader, create_naflex_loader, create_transform
from timm.data.auto_augment import (
    _HPARAMS_DEFAULT,
    AugMixAugment,
    AutoAugment,
    RandAugment,
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from timm.data.mixup import rand_bbox_minmax


@pytest.mark.parametrize('count', [None, 4])
def test_rand_bbox_minmax_can_reach_bottom_right_border(monkeypatch, count):
    def sample_last(low, high, size=None):
        result = np.asarray(high) - 1
        return result if size is None else np.broadcast_to(result, size)

    monkeypatch.setattr(np.random, 'randint', sample_last)

    yl, yu, xl, xu = rand_bbox_minmax((3, 5, 7), (0.5, 0.75), count=count)

    assert np.all(yl == 3)
    assert np.all(yu == 5)
    assert np.all(xl == 3)
    assert np.all(xu == 7)


def test_naflex_eval_patchify_can_preserve_spatial_patch_dimensions():
    common_kwargs = dict(
        input_size=(3, 16, 16),
        is_training=False,
        naflex=True,
        patch_size=(2, 2),
        max_seq_len=16,
        patchify=True,
    )
    image = torch.rand(3, 8, 8)

    spatial = create_transform(**common_kwargs, patchify_flatten=False)(image)
    flattened = create_transform(**common_kwargs)(image)

    assert spatial['patches'].ndim == 4
    assert spatial['patches'].shape[1:] == (2, 2, 3)
    assert flattened['patches'].ndim == 2
    assert flattened['patches'].shape[-1] == 2 * 2 * 3


def test_create_loader_disables_persistent_workers_without_workers():
    dataset = torch.utils.data.TensorDataset(torch.zeros((4, 3, 8, 8)), torch.arange(4))

    loader = create_loader(
        dataset,
        input_size=(3, 8, 8),
        batch_size=2,
        num_workers=0,
        persistent_workers=True,
        use_prefetcher=False,
    )

    assert loader.num_workers == 0
    assert not loader.persistent_workers


@pytest.mark.parametrize('is_training', [False, True])
def test_create_naflex_loader_disables_persistent_workers_without_workers(is_training):
    dataset = torch.utils.data.TensorDataset(torch.zeros((4, 3, 8, 8)), torch.arange(4))

    loader = create_naflex_loader(
        dataset,
        patch_size=1,
        train_seq_lens=(1,),
        max_seq_len=1,
        batch_size=2,
        is_training=is_training,
        no_aug=True,
        num_workers=0,
        persistent_workers=True,
        use_prefetcher=False,
    )

    assert loader.num_workers == 0
    assert not loader.persistent_workers


@pytest.mark.parametrize('factory, config_str, expected_type', [
    # each config string is taken verbatim from the corresponding factory's own docstring
    (auto_augment_transform, 'original-mstd0.5', AutoAugment),
    (auto_augment_transform, 'original', AutoAugment),
    (rand_augment_transform, 'rand-m9-n3-mstd0.5', RandAugment),
    (rand_augment_transform, 'rand-m9-n3-mmax8', RandAugment),
    (rand_augment_transform, 'rand-mstd1-tweights', RandAugment),
    (augment_and_mix_transform, 'augmix-m5-w4-d2', AugMixAugment),
    (augment_and_mix_transform, 'augmix', AugMixAugment),
])
def test_auto_augment_factory_accepts_default_none_hparams(factory, config_str, expected_type):
    # hparams defaults to None on all three public factories, but the mstd/mmax config sections
    # (and augmix's unconditional magnitude_std default) call hparams.setdefault(...) before it is
    # normalized, so the documented default used to raise AttributeError on a None hparams.
    before = dict(_HPARAMS_DEFAULT)
    transform = factory(config_str)
    assert isinstance(transform, expected_type)
    # the setdefault() calls must not leak magnitude_std into the shared module-level default,
    # which would otherwise pin the value for every later call that relies on the defaults.
    assert _HPARAMS_DEFAULT == before


@pytest.mark.parametrize('config_str, expected_increasing', [
    ('rand-m9-n3', False),       # flag omitted -> default off
    ('rand-m9-n3-inc0', False),  # documented off value
    ('rand-m9-n3-inc1', True),   # documented on value
])
def test_rand_augment_inc_flag_respects_zero(config_str, expected_increasing):
    # 'inc' is documented as integer(bool) with default 0 and must toggle the increasing-severity
    # transform set. val is the raw config string, and bool('0') is True, so 'inc0' used to enable it.
    transform = rand_augment_transform(config_str, hparams={})
    uses_increasing = any('Increasing' in op.name for op in transform.ops)
    assert uses_increasing == expected_increasing


@pytest.mark.parametrize('config_str, expected_blended', [
    ('augmix-m5-w4', False),     # flag omitted -> default off
    ('augmix-m5-w4-b0', False),  # documented off value
    ('augmix-m5-w4-b1', True),   # documented on value
])
def test_augmix_blended_flag_respects_zero(config_str, expected_blended):
    # 'b' (blended) is documented as integer(bool) with default 0; bool('0') is True, so 'b0' used
    # to select the blended code path instead of disabling it.
    transform = augment_and_mix_transform(config_str, hparams={})
    assert transform.blended == expected_blended
