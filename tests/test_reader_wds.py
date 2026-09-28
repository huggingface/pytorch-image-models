import io
import json
import tarfile

import pytest
import torch
from PIL import Image

from timm.data import MultiLabelTarget, create_dataset

pytest.importorskip('webdataset')

LABELS = [[5, 1, 0], [2], [], [4], [1, 5], [3], [0, 3], [2]]


def _add_member(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def wds_root(tmp_path):
    """Two shards of four samples for a 'validation' split with .cls and .json, and a 'nojson' split without."""
    image = io.BytesIO()
    Image.new('RGB', (12, 12), color=(80, 120, 160)).save(image, format='JPEG')
    splits = {}
    for split, with_json in (('validation', True), ('nojson', False)):
        filenames = []
        for shard in range(2):
            filenames.append(f'{split}-{shard:06d}.tar')
            with tarfile.open(tmp_path / filenames[-1], 'w') as tar:
                for i in range(4):
                    idx = shard * 4 + i
                    _add_member(tar, f'{idx:06d}.jpg', image.getvalue())
                    _add_member(tar, f'{idx:06d}.cls', str(idx % 6).encode())
                    if with_json:
                        meta = {'labels': LABELS[idx], 'alt': -1 if idx == 3 else idx % 2, 'cls': (idx + 1) % 6,
                                'person': idx % 2, 'dark': idx >= 4, 'output': {'pollen': float(idx % 3 == 0)},
                                'multihot': [int(c in LABELS[idx]) for c in range(6)],
                                'boxes': [{'category': c, 'area': 1} for c in LABELS[idx]]}
                        _add_member(tar, f'{idx:06d}.json', json.dumps(meta).encode())
        splits[split] = dict(name=split, num_samples=8, filenames=filenames, shard_lengths=[4, 4])
    splits['alt'] = dict(splits['validation'], name='alt', alt_label='alt')
    (tmp_path / '_info.json').write_text(json.dumps({'splits': splits}))
    return str(tmp_path)


def test_wds_cls_file_is_the_default_target(wds_root):
    dataset = create_dataset('wds/', root=wds_root, split='validation')
    assert [target for _, target in dataset] == [i % 6 for i in range(8)]
    # any explicit target key, including 'cls', comes from the json sidecar rather than the .cls file
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='cls')
    assert [target for _, target in dataset] == [(i + 1) % 6 for i in range(8)]


def test_wds_alt_label_from_info_skips_negative(wds_root):
    dataset = create_dataset('wds/', root=wds_root, split='alt')
    assert [target for _, target in dataset] == [i % 2 for i in range(8) if i != 3]


def test_wds_target_key_reads_multilabel_lists_from_json(wds_root):
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='labels')
    assert [target for _, target in dataset] == LABELS

    dataset = create_dataset(
        'wds/', root=wds_root, split='validation', target_key='labels', target_transform=MultiLabelTarget(6))
    targets = torch.stack([target for _, target in dataset])
    torch.testing.assert_close(targets, torch.stack([MultiLabelTarget(6)(labels) for labels in LABELS]))

    dataset = create_dataset(
        'wds/', root=wds_root, split='validation', target_key='labels', class_map={i: 5 - i for i in range(6)})
    assert next(iter(dataset))[1] == [0, 4, 5]


def test_wds_target_key_overrides_alt_label(wds_root):
    dataset = create_dataset('wds/', root=wds_root, split='alt', target_key='labels')
    assert [target for _, target in dataset] == LABELS


def test_wds_multi_field_binary_targets(wds_root):
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='person,dark')
    assert dataset.reader.class_to_idx == {'person': 0, 'dark': 1}
    assert [target for _, target in dataset] == [[], [0], [], [0], [1], [0, 1], [1], [0, 1]]
    dataset = create_dataset(
        'wds/', root=wds_root, split='validation', target_key='person,dark', class_map={'person': 1, 'dark': 0})
    assert [target for _, target in dataset][5] == [1, 0]
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='person,nope')
    with pytest.raises(ValueError, match='not found'):
        list(dataset)
    # '/' paths into the sidecar, for multi-field and single targets
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='output/pollen,person')
    assert [target for _, target in dataset] == [[0], [1], [], [0, 1], [], [1], [0], [1]]
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='output/pollen')
    assert [target for _, target in dataset] == [1, 0, 0, 1, 0, 0, 1, 0]
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='output/nope')
    with pytest.raises(ValueError, match='not found'):
        list(dataset)


def test_wds_multihot_target_format(wds_root):
    expected = torch.stack([MultiLabelTarget(6)(labels) for labels in LABELS])
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='multihot',
                             target_format='multihot', target_transform=MultiLabelTarget(6, dense=True))
    torch.testing.assert_close(torch.stack([target for _, target in dataset]), expected)
    # lists pass through untouched in the default indices mode, so an int multi-hot reads as indices
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='multihot')
    assert next(iter(dataset))[1] == [1, 1, 0, 0, 0, 1]
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='multihot',
                             target_format='multihot', class_map={i: 5 - i for i in range(6)})
    assert next(iter(dataset))[1] == [1., 0., 0., 0., 1., 1.]


def test_wds_path_into_json_array_of_objects(wds_root):
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='boxes/category')
    assert [target for _, target in dataset] == LABELS
    dataset = create_dataset('wds/', root=wds_root, split='validation', target_key='boxes/nope')
    with pytest.raises(ValueError, match='not found'):
        list(dataset)


@pytest.mark.parametrize('split,target_key,match', [
    ('validation', 'missing', 'not found'),
    ('nojson', 'labels', 'json sidecar'),
])
def test_wds_target_key_errors_are_not_swallowed(wds_root, split, target_key, match):
    dataset = create_dataset('wds/', root=wds_root, split=split, target_key=target_key)
    with pytest.raises(ValueError, match=match):
        list(dataset)
