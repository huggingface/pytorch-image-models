import torch

from timm.data import create_transform


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
