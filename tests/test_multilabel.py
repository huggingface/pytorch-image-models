import csv
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn

from timm.data import MultiLabelTarget, create_dataset, create_loader
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.loader import fast_collate
from timm.data.mixup import FastCollateMixup, Mixup, mixup_target
from timm.data.naflex_mixup import NaFlexMixup, pairwise_mixup_target
from timm.loss import BinaryCrossEntropy, JsdCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.task import ClassificationTask, MultiLabelClassificationTask, create_classification_loss
from timm.task.evaluator import (
    ClassificationEvaluator, MultiLabelClassificationEvaluator, evaluation_sample_limit,
)


def test_multilabel_target_encoding():
    encode = MultiLabelTarget(6)
    torch.testing.assert_close(encode([5, 1, 0, 5]), torch.tensor([1., 1., 0., 0., 0., 1.]))
    torch.testing.assert_close(encode([]), torch.zeros(6))
    torch.testing.assert_close(encode(np.array([2, 4])), torch.tensor([0., 0., 1., 0., 1., 0.]))
    # dense=True takes an already multi-hot vector of any numeric dtype and keeps soft values
    dense = torch.tensor([1., 1., 0., 0., 0., 1.])
    encode_dense = MultiLabelTarget(6, dense=True)
    torch.testing.assert_close(encode_dense(dense.bool()), dense)
    torch.testing.assert_close(encode_dense([1, 1, 0, 0, 0, 1]), dense)  # int multi-hot, ambiguous without the flag
    soft = [1., 0.5, 0., 0., 0., 1.]
    torch.testing.assert_close(encode_dense(np.array(soft)), torch.tensor(soft))
    assert encode_dense(dense.bool()).dtype == torch.float32


@pytest.mark.parametrize('target', [
    2, [-1], [6], [1.5], [True], [[1, 2]], [0.5] * 5, [2.] * 6, [-1.] * 6, [float('nan')] * 6,
])
def test_multilabel_target_rejects_invalid_indices(target):
    with pytest.raises(ValueError):
        MultiLabelTarget(6)(target)


@pytest.mark.parametrize('target', [2, [1, 0, 1], [2] * 6, [-1] * 6, [0.5] * 5, [float('nan')] * 6, [[1] * 6], []])
def test_multilabel_dense_target_rejects_invalid(target):
    with pytest.raises(ValueError):
        MultiLabelTarget(6, dense=True)(target)


def test_fast_collate_casts_scalar_targets_to_int64():
    for scalar in (np.int32(1), np.uint8(2), 3, True, np.bool_(False), 1.0, np.float32(0.)):
        _, targets = fast_collate([(np.zeros((3, 4, 4), np.uint8), scalar)] * 2)
        assert targets.dtype == torch.int64
    _, targets = fast_collate([(np.zeros((3, 4, 4), np.uint8), torch.tensor([1., 0.]))] * 2)
    assert targets.dtype == torch.float32 and targets.shape == (2, 2)


def test_class_map_remap_sequences_and_dense_targets():
    from timm.data.readers.class_map import remap_dense_target, remap_labels, remap_target

    class_map = {i: 5 - i for i in range(6)}
    assert remap_labels(np.array([2, 5]), class_map) == [3, 0]
    assert remap_labels(np.int64(2), class_map) == 3
    assert remap_labels((0, 1), {'a': 5, 'b': 4}, {0: 'a', 1: 'b'}) == [5, 4]
    assert remap_dense_target([0., 0., 1.], {0: 2, 1: 1, 2: 0}) == [1., 0., 0.]
    assert remap_dense_target(np.array([1, 0, 1, 0]), {0: 1, 2: 0}) == [1., 1.]  # unmapped positions are dropped
    assert remap_target([0, 1, 1], {0: 2, 1: 1, 2: 0}, dense=True) == [1., 1., 0.]
    assert remap_target([0, 1, 1], {0: 2, 1: 1, 2: 0}) == [2, 1, 1]
    with pytest.raises(ValueError, match='keyed by source class index'):
        remap_dense_target([0., 1.], {'a': 0, 'b': 1})
    with pytest.raises(ValueError, match='source index'):
        remap_dense_target([0., 1.], {5: 0})


@pytest.mark.parametrize('as_numpy', [False, True])
@pytest.mark.parametrize('num_splits', [1, 2])
@pytest.mark.parametrize('multi_label', [False, True])
def test_fast_collate_preserves_targets(as_numpy, num_splits, multi_label):
    targets = [torch.tensor([1., 0., 1.]), torch.tensor([0., 1., 1.])] if multi_label else [2, 1]
    batch = []
    for i, target in enumerate(targets):
        views = [torch.full((3, 4, 4), i + j * 2, dtype=torch.uint8) for j in range(num_splits)]
        if as_numpy:
            views = [view.numpy() for view in views]
        batch.append((views[0] if num_splits == 1 else tuple(views), target))
    images, actual = fast_collate(batch)
    expected = torch.stack(targets).repeat(num_splits, 1) if multi_label else torch.tensor(targets * num_splits)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(images[:, 0, 0, 0], torch.arange(2 * num_splits, dtype=torch.uint8))


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('mode', ['batch', 'pair', 'elem'])
@pytest.mark.parametrize('cutmix', [False, True])
def test_multilabel_mixup_preserves_shared_labels(monkeypatch, fast, mode, cutmix):
    import timm.data.mixup as mixup_module

    targets = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    cls = FastCollateMixup if fast else Mixup
    mix = cls(num_classes=3, multi_label=True, label_smoothing=0., mode=mode)
    monkeypatch.setattr(mix, '_params_per_batch', lambda: (0.25, cutmix))
    monkeypatch.setattr(mix, '_params_per_elem',
                        lambda n: (np.full(n, 0.25, dtype=np.float32), np.full(n, cutmix)))
    monkeypatch.setattr(mixup_module, 'cutmix_bbox_and_lam', lambda *a, **kw: ((0, 2, 0, 2), 0.75))
    images = torch.stack([torch.zeros(3, 4, 4), torch.full((3, 4, 4), 200.)])
    if fast:
        _, actual = mix(list(zip(images.byte(), targets)))
    else:
        _, actual = mix(images, targets.clone())
    lam = 0.75 if cutmix else 0.25
    torch.testing.assert_close(actual, targets * lam + targets.flip(0) * (1 - lam))
    torch.testing.assert_close(actual[:, 2], torch.ones(2))


def test_multilabel_naflex_mixup():
    targets = torch.tensor([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    actual = pairwise_mixup_target(
        targets, {0: 1, 1: 0}, [0.25, 0.75, 1.], num_classes=3, multi_label=True)
    torch.testing.assert_close(actual, torch.tensor([[0.25, 0.75, 1.], [0.25, 0.75, 1.], [1., 1., 0.]]))
    mix = NaFlexMixup(num_classes=3, multi_label=True, mixup_alpha=1., cutmix_alpha=0., mixup_off_epoch=1)
    images = [torch.randn(3, 4 + i, 4 + i) for i in range(3)]
    _, mixed = mix(images, list(targets))
    assert torch.stack(mixed).shape == targets.shape
    _, single = mix(images[:1], list(targets[:1]))
    torch.testing.assert_close(torch.stack(single), targets[:1])
    mix.set_epoch(1)
    _, unmixed = mix(images, list(targets))
    torch.testing.assert_close(torch.stack(unmixed), targets)


def test_bce_smooth_dense():
    x = torch.tensor([[2., -1.], [0.5, 0.5]])
    dense = torch.tensor([[1., 0.], [0., 1.]])
    index = torch.tensor([0, 1])
    bce = nn.functional.binary_cross_entropy_with_logits
    plain = BinaryCrossEntropy(smoothing=0.2)  # dense targets are assumed to be softened upstream
    torch.testing.assert_close(plain(x, dense), bce(x, dense))
    smooth = BinaryCrossEntropy(smoothing=0.2, smooth_dense=True)
    torch.testing.assert_close(smooth(x, dense), bce(x, dense * 0.8 + 0.1))
    torch.testing.assert_close(smooth(x, index), plain(x, index))  # index targets keep one-hot smoothing


@pytest.mark.parametrize('kwargs,expected', [
    ({}, nn.CrossEntropyLoss),
    (dict(smoothing=0.1), LabelSmoothingCrossEntropy),
    (dict(smoothing=0.1, soft_targets=True), SoftTargetCrossEntropy),
    (dict(bce=True), BinaryCrossEntropy),
    (dict(bce=True, smoothing=0.1), BinaryCrossEntropy),
    (dict(bce=True, smoothing=0.1, soft_targets=True), BinaryCrossEntropy),
    (dict(jsd_splits=2, smoothing=0.1), JsdCrossEntropy),
    (dict(multi_label=True, smoothing=0.1), BinaryCrossEntropy),
    (dict(multi_label=True, smoothing=0.1, soft_targets=True), BinaryCrossEntropy),
])
def test_create_classification_loss_dispatch(kwargs, expected):
    loss = create_classification_loss(**kwargs)
    assert isinstance(loss, expected)
    if isinstance(loss, BinaryCrossEntropy):
        assert loss.smooth_dense == kwargs.get('multi_label', False)
        assert loss.smoothing == (0. if kwargs.get('soft_targets') else kwargs.get('smoothing', 0.))


def test_classification_task_loss_kwargs_validation():
    model = nn.Linear(2, 3)
    with pytest.raises(ValueError, match='not both'):
        ClassificationTask(model, nn.CrossEntropyLoss(), smoothing=0.1, verbose=False)
    with pytest.raises(TypeError):
        ClassificationTask(model, smoothin=0.1, verbose=False)
    with pytest.raises(ValueError, match='JSD'):
        create_classification_loss(multi_label=True, jsd_splits=2)
    with pytest.raises(ValueError, match='threshold'):
        MultiLabelClassificationTask(model, threshold=1.5, verbose=False)
    assert isinstance(ClassificationTask(model, verbose=False).criterion, nn.CrossEntropyLoss)


def test_multilabel_task_creates_bce_with_dense_smoothing():
    model = nn.Linear(2, 3)
    target = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    bce = nn.functional.binary_cross_entropy_with_logits
    task = MultiLabelClassificationTask(model, smoothing=0.2, verbose=False)
    assert isinstance(task.criterion, BinaryCrossEntropy) and task.criterion.smooth_dense
    result = task(torch.ones(2, 2), target)
    torch.testing.assert_close(result['loss'], bce(result['output'], target * 0.8 + 0.1))
    result['loss'].backward()
    assert torch.isfinite(model.weight.grad).all()
    # Mixup applies the smoothing itself, so the criterion created for soft targets must not smooth again.
    mixed = mixup_target(target, 3, lam=0.25, smoothing=0.2, multi_label=True)
    torch.testing.assert_close(mixed, (target * 0.25 + target.flip(0) * 0.75) * 0.8 + 0.1)
    soft_task = MultiLabelClassificationTask(model, smoothing=0.2, soft_targets=True, verbose=False)
    result = soft_task(torch.ones(2, 2), mixed)
    torch.testing.assert_close(result['loss'], bce(result['output'], mixed))
    with pytest.raises(ValueError, match='dense floating-point'):
        task(torch.ones(2, 2), target.argmax(-1))
    task.setup_ema(decay=0.)
    task.update_ema()
    torch.testing.assert_close(task.get_eval_model()(torch.ones(2, 2)), task.get_eval_model(ema=True)(torch.ones(2, 2)))
    assert isinstance(task.create_evaluator(), MultiLabelClassificationEvaluator)


def _metric_examples():
    output = torch.tensor([[2., 2., -1.], [1., -2., 3.], [-1., 1., 0.], [1., -1., -2.]])
    target = torch.tensor([[1., 0., 0.], [1., 0., 1.], [0., 1., 1.], [0., 1., 0.]])
    return output, target


def test_multilabel_metrics_global_reduction_and_ties():
    output, target = _metric_examples()
    evaluator = MultiLabelClassificationEvaluator()
    evaluator.update(output[:1], target[:1])
    evaluator.update(output[1:], target[1:])
    metrics = evaluator.compute()
    assert metrics['loss'] == pytest.approx(nn.functional.binary_cross_entropy_with_logits(output, target).item())
    assert metrics['map'] == pytest.approx(100 * (5 / 6 + 7 / 12 + 1) / 3)
    assert metrics['micro_f1'] == pytest.approx(100 * 10 / 13)
    assert metrics['macro_f1'] == pytest.approx(100 * (0.8 + 0.5 + 1) / 3)
    assert metrics['sample_f1'] == pytest.approx(100 * (2 / 3 + 1 + 1) / 4)
    evaluator.reset()
    evaluator.update(output.flip(0), target.flip(0))
    assert evaluator.compute() == pytest.approx(metrics)
    from timm.task.evaluator import _average_precision
    torch.testing.assert_close(
        _average_precision(output, target.bool(), chunk_elements=1), _average_precision(output, target.bool()))
    with pytest.raises(ValueError, match='class indices'):
        ClassificationEvaluator().update(output, target)


def test_multilabel_metrics_empty_classes_and_threshold(caplog):
    evaluator = MultiLabelClassificationEvaluator(threshold=0.8)
    output = torch.tensor([[1., -1.], [3., -2.]])
    target = torch.tensor([[1., 0.], [1., 0.]])
    evaluator.update(output, target)
    metrics = evaluator.compute()
    assert metrics['map'] == 50.
    assert metrics['micro_f1'] == pytest.approx(200 / 3)
    assert metrics['macro_f1'] == pytest.approx(100 / 3)
    assert metrics['sample_f1'] == 50.
    assert evaluator.counts.dtype == torch.int64 and evaluator.loss_sum.dtype == torch.float64
    # soft targets are used as given for the loss and binarized at 0.5 for the ranking / F1 metrics
    soft = MultiLabelClassificationEvaluator(threshold=0.8)
    with caplog.at_level(logging.WARNING):
        soft.update(output, target * 0.6)
    soft_metrics = soft.compute()
    assert 'binarizes soft targets' in caplog.text
    for name in ('map', 'micro_f1', 'macro_f1', 'sample_f1'):
        assert soft_metrics[name] == pytest.approx(metrics[name])
    assert soft_metrics['loss'] == pytest.approx(
        nn.functional.binary_cross_entropy_with_logits(output, target * 0.6).item())


@pytest.mark.parametrize('evaluator_cls', [ClassificationEvaluator, MultiLabelClassificationEvaluator])
def test_evaluator_metric_names_match_compute(evaluator_cls):
    output, target = _metric_examples()
    if evaluator_cls is ClassificationEvaluator:
        target = target.argmax(-1)
    evaluator = evaluator_cls()
    evaluator.update(output, target)
    assert tuple(evaluator.compute()) == evaluator.metric_names
    assert tuple(evaluator.summary()) == evaluator.metric_names[:len(evaluator.summary())]
    assert evaluator.default_metric in evaluator.metric_names


def test_single_label_evaluator_matches_accuracy_and_handles_small_head():
    output, target = _metric_examples()
    target = target.argmax(-1)
    evaluator = ClassificationTask(nn.Identity(), nn.CrossEntropyLoss(), verbose=False).create_evaluator()
    evaluator.update(output[:3], target[:3])
    evaluator.update(output[3:], target[3:])
    metrics = evaluator.compute()
    assert metrics['loss'] == pytest.approx(nn.functional.cross_entropy(output, target).item())
    assert metrics['top1'] == 50.
    assert metrics['top5'] == 100.


def _distributed_metrics_worker(rank, init_file, output_dir, size, multi_label):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2)
    try:
        output, target = _metric_examples()
        if not multi_label:
            target = target.argmax(-1)
        sampler = OrderedDistributedSampler(list(range(size)), num_replicas=2, rank=rank)
        evaluator_cls = MultiLabelClassificationEvaluator if multi_label else ClassificationEvaluator
        evaluator = evaluator_cls(max_samples=evaluation_sample_limit(SimpleNamespace(sampler=sampler)))
        indices = list(sampler)
        evaluator.update(output[indices], target[indices])
        metrics = evaluator.compute(distributed=True)
        Path(output_dir, f'{rank}.json').write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='requires Gloo')
@pytest.mark.parametrize('size', [1, 3])
@pytest.mark.parametrize('multi_label', [False, True])
def test_distributed_metrics_exclude_padding(tmp_path, size, multi_label):
    torch.multiprocessing.spawn(
        _distributed_metrics_worker,
        args=(str(tmp_path / 'init'), str(tmp_path), size, multi_label),
        nprocs=2,
        join=True,
    )
    output, target = _metric_examples()
    if not multi_label:
        target = target.argmax(-1)
    evaluator = MultiLabelClassificationEvaluator() if multi_label else ClassificationEvaluator()
    evaluator.update(output[:size], target[:size])
    expected = evaluator.compute()
    for rank in range(2):
        assert json.loads((tmp_path / f'{rank}.json').read_text()) == pytest.approx(expected)


_FIXTURE_LABELS = ([5, 1, 0], [2], [], [4], [1, 5], [3], [0, 3], [2])


@pytest.fixture
def hf_dataset(monkeypatch):
    datasets = pytest.importorskip('datasets')
    data = io.BytesIO()
    Image.new('RGB', (12, 12), color=(80, 120, 160)).save(data, format='PNG')
    names = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
    features = datasets.Features({
        'image': datasets.Image(),
        'labels': datasets.Sequence(datasets.ClassLabel(names=names)),
        # one binary field per label, in a mix of feature types
        'person': datasets.ClassLabel(names=['no', 'yes']),
        'dark': datasets.Value('bool'),
        'far': datasets.Value('int64'),
        # nested, bee_dataset style
        'output': {'pollen': datasets.Value('float64'), 'wasps': datasets.Value('float64')},
        'multihot': datasets.Sequence(datasets.Value('int64')),  # dense multi-hot with int values
    })
    info = datasets.DatasetInfo(features=features, splits=datasets.SplitDict({
        name: datasets.SplitInfo(name=name, num_examples=8) for name in ('train', 'validation')
    }))
    dataset = datasets.Dataset.from_dict({
        'image': [{'bytes': data.getvalue(), 'path': f'{i}.png'} for i in range(8)],
        'labels': list(_FIXTURE_LABELS),
        'multihot': [[int(c in labels) for c in range(6)] for labels in _FIXTURE_LABELS],
        'person': [1, 0, 1, 0, 1, 0, 1, 0],
        'dark': [False, False, True, True, False, False, True, True],
        'far': [0, 1, 0, 1, 0, 1, 0, 1],
        'output': [{'pollen': float(i % 3 == 0), 'wasps': float(i >= 6)} for i in range(8)],
    }, info=info)
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: dataset)
    builder = SimpleNamespace(
        info=dataset.info,
        as_streaming_dataset=lambda **kw: dataset.to_iterable_dataset(),
        as_dataset=lambda **kw: dataset,
        download_and_prepare=lambda: None,
    )
    monkeypatch.setattr(datasets, 'load_dataset_builder', lambda *a, **kw: builder)
    return dataset


@pytest.mark.parametrize('reader', ['hfds', 'hfids'])
@pytest.mark.parametrize('prefetch', [False, True])
def test_hf_multilabel_readers_and_loader(hf_dataset, reader, prefetch):
    dataset = create_dataset(
        f'{reader}/fixture', split='validation', target_key='labels', target_transform=MultiLabelTarget(6))
    assert dataset.reader.class_to_idx['scab'] == 5
    loader = create_loader(dataset, input_size=(3, 8, 8), batch_size=3, num_workers=0,
                           device=torch.device('cpu'), use_prefetcher=prefetch)
    batches = list(loader)
    assert len(loader) == len(batches)
    assert sum(len(target) for _, target in batches) == 8
    torch.testing.assert_close(batches[0][1][0], torch.tensor([1., 1., 0., 0., 0., 1.]))
    assert batches[0][1].dtype == torch.float32


def test_hfids_more_workers_than_shards_fails_fast_when_training(hf_dataset):
    """A worker with no shard yields nothing; training must not loop on that forever (datasets>=5 shuffle reports 1)."""
    dataset = create_dataset('hfids/fixture', split='train', target_key='labels', target_transform=MultiLabelTarget(6),
                             is_training=True, batch_size=2, num_samples=8)
    loader = create_loader(dataset, input_size=(3, 8, 8), batch_size=2, is_training=True, num_workers=2,
                           device=torch.device('cpu'), use_prefetcher=False, no_aug=True)
    with pytest.raises(RuntimeError, match='would receive no data'):
        next(iter(loader))
    dataset = create_dataset('hfids/fixture', split='validation', target_key='labels', target_transform=MultiLabelTarget(6))
    loader = create_loader(dataset, input_size=(3, 8, 8), batch_size=2, num_workers=2, device=torch.device('cpu'),
                           use_prefetcher=False)
    assert sum(len(target) for _, target in loader) == 8  # eval: the extra worker idles (warning logged in the worker)


@pytest.mark.parametrize('reader', ['hfds', 'hfids'])
@pytest.mark.parametrize('named_map', [False, True])
def test_hf_multilabel_class_map(hf_dataset, reader, named_map):
    names = hf_dataset.features['labels'].feature.names
    class_map = {name if named_map else i: 5 - i for i, name in enumerate(names)}
    dataset = create_dataset(f'{reader}/fixture', split='validation', target_key='labels', class_map=class_map)
    assert next(iter(dataset))[1] == [0, 4, 5]


@pytest.mark.parametrize('reader', ['hfds', 'hfids'])
def test_hf_multi_field_binary_targets(hf_dataset, reader):
    keys = 'person,far,dark'
    dataset = create_dataset(f'{reader}/fixture', split='validation', target_key=keys)
    assert dataset.reader.class_to_idx == {'person': 0, 'far': 1, 'dark': 2}
    assert [target for _, target in dataset] == [[0], [1], [0, 2], [1, 2], [0], [1], [0, 2], [1, 2]]
    dataset = create_dataset(
        f'{reader}/fixture', split='validation', target_key=keys, target_transform=MultiLabelTarget(3))
    torch.testing.assert_close(next(iter(dataset))[1], torch.tensor([1., 0., 0.]))
    dataset = create_dataset(
        f'{reader}/fixture', split='validation', target_key=keys, class_map={'person': 2, 'far': 1, 'dark': 0})
    assert [target for _, target in dataset][:3] == [[2], [1], [2, 0]]
    with pytest.raises(ValueError, match='not found in dataset features'):
        create_dataset(f'{reader}/fixture', split='validation', target_key='person,nope')
    # '/' paths address fields nested in sub-dicts, for multi-field and single targets alike
    dataset = create_dataset(f'{reader}/fixture', split='validation', target_key='output/pollen,person,output/wasps')
    assert dataset.reader.class_to_idx == {'output/pollen': 0, 'person': 1, 'output/wasps': 2}
    assert [target for _, target in dataset] == [[0, 1], [], [1], [0], [1], [], [0, 1, 2], [2]]
    dataset = create_dataset(f'{reader}/fixture', split='validation', target_key='output/wasps')
    assert [target for _, target in dataset] == [0., 0., 0., 0., 0., 0., 1., 1.]
    with pytest.raises(ValueError, match='not found in dataset features'):
        create_dataset(f'{reader}/fixture', split='validation', target_key='person,output/nope')
    with pytest.raises(ValueError, match='unique'):
        create_dataset(f'{reader}/fixture', split='validation', target_key='person,person')


@pytest.mark.parametrize('reader', ['hfds', 'hfids'])
@pytest.mark.parametrize('kind', ['Sequence', 'List'])
def test_hf_sequence_of_dicts_path_yields_label_list(monkeypatch, reader, kind):
    """Detection style 'objects/category': Sequence stores a dict of lists, List a list of dicts."""
    datasets = pytest.importorskip('datasets')
    labels = datasets.ClassLabel(names=['coverall', 'gloves', 'mask'])
    fields = {'area': datasets.Value('int64'), 'category': labels}
    per_sample = [([1, 2], [2, 0]), ([], []), ([3], [1])]
    if kind == 'Sequence':
        objects_feature = datasets.Sequence(fields)
        objects = [{'area': areas, 'category': cats} for areas, cats in per_sample]
    else:
        objects_feature = datasets.List(fields)
        objects = [[{'area': a, 'category': c} for a, c in zip(areas, cats)] for areas, cats in per_sample]
    features = datasets.Features({'image': datasets.Image(), 'objects': objects_feature})
    data = io.BytesIO()
    Image.new('RGB', (8, 8)).save(data, format='PNG')
    hf_dataset = datasets.Dataset.from_dict({
        'image': [{'bytes': data.getvalue(), 'path': f'{i}.png'} for i in range(3)],
        'objects': objects,
    }, info=datasets.DatasetInfo(features=features, splits=datasets.SplitDict({
        'train': datasets.SplitInfo(name='train', num_examples=3)})))
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: hf_dataset)
    monkeypatch.setattr(datasets, 'load_dataset_builder', lambda *a, **kw: SimpleNamespace(
        info=hf_dataset.info, as_streaming_dataset=lambda **kw: hf_dataset.to_iterable_dataset(),
        as_dataset=lambda **kw: hf_dataset, download_and_prepare=lambda: None))
    name = f'{reader}/fixture'
    dataset = create_dataset(name, split='train', target_key='objects/category')
    assert dataset.reader.class_to_idx == {'coverall': 0, 'gloves': 1, 'mask': 2}
    assert [target for _, target in dataset] == [[2, 0], [], [1]]
    dataset = create_dataset(
        name, split='train', target_key='objects/category', target_transform=MultiLabelTarget(3))
    torch.testing.assert_close(torch.stack([target for _, target in dataset]),
                               torch.tensor([[1., 0., 1.], [0., 0., 0.], [0., 1., 0.]]))
    dataset = create_dataset(name, split='train', target_key='objects/category',
                             class_map={'coverall': 2, 'gloves': 1, 'mask': 0})
    assert next(iter(dataset))[1] == [0, 2]


def test_hf_remote_code_kwargs():
    from timm.data.readers._hfds import remote_code_kwargs

    assert remote_code_kwargs(False) == {}
    datasets = pytest.importorskip('datasets')
    import inspect
    if 'trust_remote_code' in inspect.signature(datasets.load_dataset).parameters:
        assert remote_code_kwargs(True) == {'trust_remote_code': True}
    else:
        with pytest.raises(ValueError, match='datasets 4.0'):
            remote_code_kwargs(True)


@pytest.mark.parametrize('reader', ['hfds', 'hfids'])
def test_hf_multihot_target_format(hf_dataset, reader):
    expected = torch.stack([MultiLabelTarget(6)(labels) for labels in _FIXTURE_LABELS])
    dense = MultiLabelTarget(6, dense=True)
    dataset = create_dataset(
        f'{reader}/fixture', split='validation', target_key='multihot', target_format='multihot',
        target_transform=dense)
    torch.testing.assert_close(torch.stack([target for _, target in dataset]), expected)
    # without the format flag an int multi-hot list is read as class indices, by design
    dataset = create_dataset(f'{reader}/fixture', split='validation', target_key='multihot')
    assert next(iter(dataset))[1] == [1, 1, 0, 0, 0, 1]
    # class maps remap dense targets by position and must be keyed by source index
    dataset = create_dataset(f'{reader}/fixture', split='validation', target_key='multihot', target_format='multihot',
                             class_map={i: 5 - i for i in range(6)}, target_transform=dense)
    torch.testing.assert_close(next(iter(dataset))[1], expected[0].flip(0))
    dataset = create_dataset(f'{reader}/fixture', split='validation', target_key='multihot', target_format='multihot',
                             class_map={'a': 0})
    with pytest.raises(ValueError, match='keyed by source class index'):
        next(iter(dataset))
    with pytest.raises(ValueError, match='own target format'):
        create_dataset(f'{reader}/fixture', split='validation', target_key='person,dark', target_format='multihot')
    with pytest.raises(ValueError, match='Unknown target format'):
        create_dataset(f'{reader}/fixture', split='validation', target_key='multihot', target_format='dense')


def test_hf_class_metadata_with_custom_column():
    datasets = pytest.importorskip('datasets')
    from timm.data.readers._hfds import get_class_labels

    labels = datasets.ClassLabel(names=['a', 'b'])
    for feature in (labels, datasets.Sequence(labels)):
        info = datasets.DatasetInfo(features=datasets.Features({'custom': feature}))
        assert get_class_labels(info, 'custom') == {'a': 0, 'b': 1}
        assert get_class_labels(info) == {}
    assert get_class_labels(datasets.DatasetInfo()) == {}
    info = datasets.DatasetInfo(features=datasets.Features({'meta': {'cls': labels}}))
    assert get_class_labels(info, 'meta/cls') == {'a': 0, 'b': 1}
    assert get_class_labels(info, 'meta/nope') == {}
    # detection style: a ClassLabel inside a Sequence of dicts (e.g. cppe-5 'objects/category')
    objects = datasets.Sequence({'bbox': datasets.Sequence(datasets.Value('float32')), 'category': labels})
    info = datasets.DatasetInfo(features=datasets.Features({'objects': objects}))
    assert get_class_labels(info, 'objects/category') == {'a': 0, 'b': 1}
    assert get_class_labels(info, 'objects/nope') == {}


def test_target_field_paths_prefer_literal_keys():
    from timm.data.readers.targets import get_field, has_field, multi_field_target

    sample = {'a/b': 1, 'a': {'b': 2, 'c': {'d': 0}}, 'e': 3}
    assert get_field(sample, 'a/b') == 1 and get_field(sample, 'a/c/d') == 0 and get_field(sample, 'e') == 3
    assert has_field(sample, 'a/c') and not has_field(sample, 'a/x') and not has_field(sample, 'e/x')
    assert multi_field_target(sample, ('a/c/d', 'a/b', 'e')) == [1, 2]
    with pytest.raises(KeyError):
        get_field(sample, 'a/x')
    # a list of dicts (HF List of dicts, JSON arrays of objects) maps the remaining path over its items
    boxes = {'objects': [{'category': 0, 'tag': {'v': 1}}, {'category': 2, 'tag': {'v': 3}}], 'empty': []}
    assert get_field(boxes, 'objects/category') == [0, 2]
    assert get_field(boxes, 'objects/tag/v') == [1, 3]
    assert get_field(boxes, 'empty/category') == []
    assert has_field(boxes, 'objects/category') and has_field(boxes, 'empty/category')
    assert not has_field(boxes, 'objects/nope')
    assert not has_field({'objects': [{'category': 0}, {}]}, 'objects/category')  # every item needs the key


@pytest.mark.parametrize('prefetch', [False, True])
@pytest.mark.parametrize('is_training', [False, True])
def test_naflex_multilabel_loader(hf_dataset, prefetch, is_training):
    from timm.data import create_naflex_loader

    dataset = create_dataset('hfds/fixture', target_key='labels', target_transform=MultiLabelTarget(6))
    mixup = NaFlexMixup(num_classes=6, multi_label=True, mixup_alpha=1., cutmix_alpha=0.)
    loader = create_naflex_loader(
        dataset, patch_size=4, train_seq_lens=(4, 9), max_seq_len=9, batch_size=8,
        is_training=is_training, mixup_fn=mixup if is_training else None, no_aug=True,
        num_workers=0, use_prefetcher=prefetch, device=torch.device('cpu'),
    )
    batches = list(loader)
    assert batches
    for _, targets in batches:
        assert targets.ndim == 2 and targets.shape[1] == 6
        assert targets.dtype == torch.float32
        assert torch.all((targets >= 0) & (targets <= 1))
    if not is_training:
        torch.testing.assert_close(batches[0][1][0], MultiLabelTarget(6)([5, 1, 0]))


def test_train_cli_rejects_metric_the_evaluator_does_not_produce(monkeypatch, tmp_path, hf_dataset):
    import train

    monkeypatch.setattr(train, 'create_model', lambda *a, **kw: _TinyClassifier(kw['num_classes']))
    monkeypatch.setattr('sys.argv', [
        'train.py', '--task', 'multilabel', '--dataset', 'hfds/fixture', '--target-key', 'labels',
        '--num-classes', '6', '--device', 'cpu', '--workers', '0', '--batch-size', '2', '--img-size', '8',
        '--epochs', '1', '--eval-metric', 'top1', '--output', str(tmp_path),
    ])
    with pytest.raises(ValueError, match='eval-metric'):
        train.main()


def test_multilabel_train_cli_multihot(monkeypatch, tmp_path, hf_dataset):
    import train

    monkeypatch.setattr(train, 'create_model', lambda *a, **kw: _TinyClassifier(kw['num_classes']))
    common = ['--dataset', 'hfds/fixture', '--target-key', 'multihot', '--target-format', 'multihot',
              '--num-classes', '6', '--device', 'cpu', '--workers', '0', '--batch-size', '2', '--img-size', '8']
    monkeypatch.setattr('sys.argv', ['train.py', '--task', 'classification', *common])
    with pytest.raises(SystemExit):
        train._parse_args()
    monkeypatch.setattr('sys.argv', [
        'train.py', '--task', 'multilabel', *common, '--epochs', '1', '--warmup-epochs', '0', '--opt', 'sgd',
        '--lr', '0.01', '--no-aug', '--output', str(tmp_path), '--experiment', 'multihot',
    ])
    train.main()
    summary = list(csv.DictReader((tmp_path / 'multihot' / 'summary.csv').open()))
    assert 0. <= float(summary[0]['eval_map']) <= 100.


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(3, num_classes)
        self.pretrained_cfg = dict(input_size=(3, 8, 8), mean=(0.5,) * 3, std=(0.5,) * 3,
                                   interpolation='bilinear', crop_pct=1.)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


@pytest.mark.parametrize('reader,prefetch,device', [
    ('hfds', True, 'cpu'), ('hfds', False, 'cpu'), ('hfids', True, 'cpu'),
    pytest.param('hfds', True, 'cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA')),
])
def test_multilabel_train_and_validate_cli(monkeypatch, tmp_path, hf_dataset, reader, prefetch, device):
    import train
    import validate

    monkeypatch.setattr(train, 'create_model', lambda *a, **kw: _TinyClassifier(kw['num_classes']))
    monkeypatch.setattr(validate, 'create_model', lambda *a, **kw: _TinyClassifier(kw['num_classes']))
    args = [
        '--task', 'multilabel', '--dataset', f'{reader}/fixture', '--target-key', 'labels',
        '--num-classes', '6', '--device', device, '--workers', '0', '--batch-size', '2', '--img-size', '8',
    ]
    if device == 'cuda':
        args.append('--amp')
    if not prefetch:
        args.append('--no-prefetcher')
    monkeypatch.setattr('sys.argv', ['train.py', *args, '--epochs', '1', '--warmup-epochs', '0',
                                    '--opt', 'sgd', '--lr', '0.01', '--no-aug', '--mixup', '0.4', '--smoothing', '0.1',
                                    '--model-ema', '--output', str(tmp_path), '--experiment', 'smoke'])
    train.main()
    summary = list(csv.DictReader((tmp_path / 'smoke' / 'summary.csv').open()))
    assert len(summary) == 1
    for name in ('map', 'micro_f1', 'macro_f1', 'sample_f1'):
        assert 0. <= float(summary[0][f'eval_{name}']) <= 100.
    assert 'eval_top1' not in summary[0]
    checkpoint = tmp_path / 'smoke' / 'model_best.pth.tar'
    assert checkpoint.is_file()
    val_args = validate.parser.parse_args([*args, '--checkpoint', str(checkpoint), '--use-ema'])
    results = validate.validate(val_args)
    assert 'map' in results and 'top1' not in results
    assert results['map'] == pytest.approx(float(summary[0]['eval_map']), abs=1e-3)
