import pytest

from timm.data import CustomDatasetInfo, DatasetInfoLabelMapper, LabelMappingCoverage


@pytest.mark.parametrize('label_names', [
    ['cat', 'dog'],
    ('cat', 'dog'),
])
def test_custom_dataset_info_sequence_labels(label_names):
    info = CustomDatasetInfo(
        label_names,
        label_descriptions={'cat': 'A cat', 'dog': 'A dog'},
    )

    assert info.num_classes() == 2
    assert info.index_to_label_name(0) == 'cat'
    assert info.index_to_description(1) == 'A dog'


def test_custom_dataset_info_sparse_json_mapping():
    info = CustomDatasetInfo(
        {'0': 'cat', '2': 'dog'},
        label_descriptions={'cat': 'A cat', 'dog': 'A dog'},
    )

    assert info.num_classes() == 2
    assert info.label_names() == {'0': 'cat', '2': 'dog'}
    assert info.label_indices() == (0, 2)
    assert info.has_label(0)
    assert info.has_label(2)
    assert not info.has_label(1)
    assert info.index_to_label_name(0) == 'cat'
    assert info.index_to_label_name(2) == 'dog'
    assert info.index_to_description(2) == 'A dog'
    with pytest.raises(KeyError):
        info.index_to_label_name(1)


def test_custom_dataset_info_mapping_validation():
    with pytest.raises(ValueError, match='integer-like'):
        CustomDatasetInfo({'cat': 'cat'})
    with pytest.raises(ValueError, match='Duplicate label index'):
        CustomDatasetInfo({'01': 'cat', 1: 'dog'})
    with pytest.raises(ValueError, match='Missing descriptions'):
        CustomDatasetInfo({0: 'cat'}, label_descriptions={'dog': 'A dog'})


def test_dataset_info_label_mapper_sparse_fallback_and_coverage():
    info = CustomDatasetInfo(
        {'0': 'cat', '2': 'dog', '4': 'horse'},
        label_descriptions={'cat': 'A cat', 'dog': 'A dog', 'horse': 'A horse'},
    )
    mapper = DatasetInfoLabelMapper(info, label_type='description')

    assert mapper(0) == 'A cat'
    assert mapper(1) == '<unmapped:1>'
    assert mapper(2) == 'A dog'
    assert mapper.coverage(4) == LabelMappingCoverage(mapped=2, missing=2, extra=1)


def test_dataset_info_label_mapper_options():
    info = CustomDatasetInfo(
        {0: 'cat'},
        label_descriptions={'cat': 'A detailed cat'},
    )

    assert DatasetInfoLabelMapper(info, label_type='name')(0) == 'cat'
    assert DatasetInfoLabelMapper(info, label_type='detailed')(0) == 'A detailed cat'
    with pytest.raises(KeyError):
        DatasetInfoLabelMapper(info, fallback_format=None)(1)
    with pytest.raises(ValueError, match='Invalid label type'):
        DatasetInfoLabelMapper(info, label_type='invalid')
