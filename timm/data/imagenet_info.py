import csv
import os
import pkgutil
import re
from typing import Dict, List, Optional, Union

from .dataset_info import DatasetInfo


# NOTE no ambiguity wrt to mapping from # classes to ImageNet subset so far, but likely to change
_NUM_CLASSES_TO_SUBSET = {
    1000: 'imagenet-1k',
    11221: 'imagenet-21k-miil',  # miil subset of fall11
    11821: 'imagenet-12k',  # timm specific 12k subset of fall11
    21841: 'imagenet-22k',  # as in fall11.tar
    21842: 'imagenet-22k-ms',  # a Microsoft (for FocalNet) remapping of 22k w/ moves ImageNet-1k classes to first 1000
    21843: 'imagenet-21k-goog',  # Google's ImageNet full has two classes not in fall11
}

_SUBSETS = {
    'imagenet1k': 'imagenet_synsets.txt',
    'imagenet12k': 'imagenet12k_synsets.txt',
    'imagenet22k': 'imagenet22k_synsets.txt',
    'imagenet21k': 'imagenet21k_goog_synsets.txt',
    'imagenet21kgoog': 'imagenet21k_goog_synsets.txt',
    'imagenet21kmiil': 'imagenet21k_miil_synsets.txt',
    'imagenet22kms': 'imagenet22k_ms_synsets.txt',
}
_LEMMA_FILE = 'imagenet_synset_to_lemma.txt'
_DEFINITION_FILE = 'imagenet_synset_to_definition.txt'


def infer_imagenet_subset(model_or_cfg) -> Optional[str]:
    if isinstance(model_or_cfg, dict):
        num_classes = model_or_cfg.get('num_classes', None)
    else:
        num_classes = getattr(model_or_cfg, 'num_classes', None)
        if not num_classes:
            pretrained_cfg = getattr(model_or_cfg, 'pretrained_cfg', {})
            # FIXME at some point pretrained_cfg should include dataset-tag,
            # which will be more robust than a guess based on num_classes
            num_classes = pretrained_cfg.get('num_classes', None)
    if not num_classes or num_classes not in _NUM_CLASSES_TO_SUBSET:
        return None
    return _NUM_CLASSES_TO_SUBSET[num_classes]


class ImageNetInfo(DatasetInfo):

    def __init__(self, subset: str = 'imagenet-1k'):
        super().__init__()
        subset = re.sub(r'[-_\s]', '', subset.lower())
        assert subset in _SUBSETS, f'Unknown imagenet subset {subset}.'

        # WordNet synsets (part-of-speach + offset) are the unique class label names for ImageNet classifiers
        synset_file = _SUBSETS[subset]
        synset_data = pkgutil.get_data(__name__, os.path.join('_info', synset_file))
        self._synsets = synset_data.decode('utf-8').splitlines()

        # WordNet lemmas (canonical dictionary form of word) and definitions are used to build
        # the class descriptions. If detailed=True both are used, otherwise just the lemmas.
        lemma_data = pkgutil.get_data(__name__, os.path.join('_info', _LEMMA_FILE))
        reader = csv.reader(lemma_data.decode('utf-8').splitlines(), delimiter='\t')
        self._lemmas = dict(reader)
        definition_data = pkgutil.get_data(__name__, os.path.join('_info', _DEFINITION_FILE))
        reader = csv.reader(definition_data.decode('utf-8').splitlines(), delimiter='\t')
        self._definitions = dict(reader)

    def num_classes(self):
        return len(self._synsets)

    def label_names(self):
        return self._synsets

    def label_descriptions(self, detailed: bool = False, as_dict: bool = False) -> Union[List[str], Dict[str, str]]:
        if as_dict:
            return {label: self.label_name_to_description(label, detailed=detailed) for label in self._synsets}
        else:
            return [self.label_name_to_description(label, detailed=detailed) for label in self._synsets]

    def index_to_label_name(self, index) -> str:
        assert 0 <= index < len(self._synsets), \
            f'Index ({index}) out of range for dataset with {len(self._synsets)} classes.'
        return self._synsets[index]

    def index_to_description(self, index: int, detailed: bool = False) -> str:
        label = self.index_to_label_name(index)
        return self.label_name_to_description(label, detailed=detailed)

    def label_name_to_description(self, label: str, detailed: bool = False) -> str:
        if detailed:
            description = f'{self._lemmas[label]}: {self._definitions[label]}'
        else:
            description = f'{self._lemmas[label]}'
        return description
