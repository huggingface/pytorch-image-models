import csv
import math
import warnings
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from timm import create_model
from timm.data import ImageDataset, NaFlexMapDatasetWrapper, NaFlexMixup, create_naflex_loader
from timm.task.evaluator import ClassificationEvaluator


class _TensorImageDataset(Dataset):
    def __init__(self, length=32):
        self.length = length

    def __getitem__(self, index):
        return torch.full((3, 2, 2), index, dtype=torch.float32), index

    def __len__(self):
        return self.length


def _create_naflex_dataset(epoch=0, mixup_fn=None):
    return NaFlexMapDatasetWrapper(
        _TensorImageDataset(),
        patch_size=1,
        seq_lens=(1,),
        max_tokens_per_batch=4,
        seed=17,
        epoch=epoch,
        batch_divisor=1,
        mixup_fn=mixup_fn,
    )


def _loader_indices(loader):
    return [index for _, targets in loader for index in targets.tolist()]


def _loader_targets(loader):
    return torch.cat([targets for _, targets in loader])


def test_naflex_epoch_batches_are_prepared_when_iteration_starts():
    dataset = _create_naflex_dataset()
    prepare_epoch_batches = dataset._prepare_epoch_batches
    prepared_epochs = []

    def prepare(epoch):
        prepared_epochs.append(epoch)
        return prepare_epoch_batches(epoch)

    dataset._prepare_epoch_batches = prepare

    assert prepared_epochs == []
    dataset.set_epoch(3)
    assert prepared_epochs == []
    assert dataset.shared_epoch.value == 3

    iterator = iter(dataset)
    assert prepared_epochs == []
    next(iterator)
    assert prepared_epochs == [3]


def test_naflex_persistent_workers_read_shared_epoch():
    dataset = _create_naflex_dataset()
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        persistent_workers=True,
    )

    try:
        epoch_0_indices = _loader_indices(loader)
        dataset.set_epoch(1)
        epoch_1_indices = _loader_indices(loader)

        expected_epoch_0 = [index for _, _, indices in dataset._prepare_epoch_batches(0) for index in indices]
        expected_epoch_1 = [index for _, _, indices in dataset._prepare_epoch_batches(1) for index in indices]
        assert epoch_0_indices == expected_epoch_0
        assert epoch_1_indices == expected_epoch_1
        assert epoch_1_indices != epoch_0_indices
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_naflex_persistent_workers_disable_mixup_at_configured_epoch():
    mixup = NaFlexMixup(
        num_classes=32,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        label_smoothing=0.0,
        mixup_off_epoch=1,
    )
    dataset = _create_naflex_dataset(mixup_fn=mixup)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        persistent_workers=True,
    )

    try:
        mixed_targets = _loader_targets(loader)
        dataset.set_epoch(1)
        unmixed_targets = _loader_targets(loader)

        assert torch.any((mixed_targets > 0) & (mixed_targets < 1))
        assert torch.all((unmixed_targets == 0) | (unmixed_targets == 1))
        torch.testing.assert_close(unmixed_targets.sum(dim=1), torch.ones(len(unmixed_targets)))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_naflex_epoch_prep_warnings_only_emitted_by_worker_zero(monkeypatch):
    dataset = _create_naflex_dataset()

    def prepare(_epoch):
        warnings.warn('epoch prep mismatch')
        return []

    dataset._prepare_epoch_batches = prepare

    for worker_id, expected_warnings in ((0, 1), (1, 0)):
        worker_info = SimpleNamespace(id=worker_id, num_workers=2)
        monkeypatch.setattr(torch.utils.data, 'get_worker_info', lambda: worker_info)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            list(dataset)
        assert len(caught) == expected_warnings


def _create_image_folder(tmp_path):
    root = tmp_path / 'images'
    for class_id in range(2):
        folder = root / str(class_id)
        folder.mkdir(parents=True)
        Image.new('RGB', (32, 32), color=(64 + 64 * class_id,) * 3).save(folder / 'image.png')
    return root


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')),
])
def test_naflex_validate_without_prefetch_preserves_metadata(tmp_path, device, dtype):
    import train

    loader = create_naflex_loader(
        ImageDataset(str(_create_image_folder(tmp_path))), patch_size=16, max_seq_len=4,
        batch_size=2, num_workers=0, use_prefetcher=False,
    )
    inputs, targets = next(iter(loader))
    original = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    model = create_model(
        'naflexvit_base_patch16_gap', pretrained=False, embed_dim=32, depth=1, num_heads=4, num_classes=2,
    ).to(device=device, dtype=dtype)

    def check_input(_model, args):
        actual = args[0]
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                expected_dtype = dtype if key == 'patches' else value.dtype
                assert actual[key].dtype == expected_dtype
                assert actual[key].device.type == device
                torch.testing.assert_close(actual[key], value.to(device=device, dtype=expected_dtype))
            else:
                assert actual[key] == value

    model.register_forward_pre_hook(check_input)
    args = SimpleNamespace(prefetcher=False, channels_last=False, tta=0, log_interval=1, distributed=False, rank=0)
    evaluator = ClassificationEvaluator(device=device)
    metrics = [train.validate(
        model, [(inputs, targets)], evaluator, args, device=torch.device(device), model_dtype=dtype,
    ) for _ in range(2)]
    assert metrics[0] == metrics[1]
    assert math.isfinite(metrics[0]['loss'])
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(inputs[key], value)
        else:
            assert inputs[key] == value


@pytest.mark.parametrize('naflex', [False, True])
@pytest.mark.parametrize('prefetch', [False, True])
@pytest.mark.parametrize('channels_last', [False, True])
@pytest.mark.usefixtures('isolate_cli_backend_flags')
def test_naflex_train_and_validate_cli(monkeypatch, tmp_path, naflex, prefetch, channels_last):
    import train
    import validate

    monkeypatch.setattr(torch.backends.cudnn, 'enabled', False, raising=False)
    args = [
        '--data-dir', str(_create_image_folder(tmp_path)), '--model', 'naflexvit_base_patch16_gap',
        '--model-kwargs', 'embed_dim=32', 'depth=1', 'num_heads=4', '--num-classes', '2',
        '--device', 'cpu', '--workers', '0', '--batch-size', '2', '--img-size', '32',
    ]
    if naflex:
        args += ['--naflex-loader', '--naflex-max-seq-len', '4']
    if not prefetch:
        args.append('--no-prefetcher')
    if channels_last:
        args.append('--channels-last')
    monkeypatch.setattr('sys.argv', [
        'train.py', *args, '--naflex-train-seq-lens', '4', '--epochs', '1', '--warmup-epochs', '0',
        '--opt', 'sgd', '--lr', '0.01', '--no-aug', '--output', str(tmp_path), '--experiment', 'smoke',
    ])
    train.main()
    with (tmp_path / 'smoke' / 'summary.csv').open() as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 1
    assert math.isfinite(float(summary[0]['train_loss']))
    assert math.isfinite(float(summary[0]['eval_loss']))
    checkpoint = tmp_path / 'smoke' / 'model_best.pth.tar'
    assert checkpoint.is_file()
    val_args = validate.parser.parse_args([*args, '--checkpoint', str(checkpoint)])
    results = validate.validate(val_args)
    assert results['top1'] == pytest.approx(float(summary[0]['eval_top1']), abs=1e-3)
