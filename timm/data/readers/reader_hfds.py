""" Dataset reader that wraps Hugging Face datasets

Hacked together by / Copyright 2022 Ross Wightman
"""
import io
from typing import List, Optional

try:
    import datasets
except ImportError as e:
    print("Please install Hugging Face datasets package `pip install datasets`.")
    raise e
from .class_map import load_class_map, remap_target
from .reader import Reader
from ._hfds import get_class_labels, remote_code_kwargs
from .targets import check_target_format, get_field, multi_field_class_to_idx, multi_field_target, parse_target_keys


class ReaderHfds(Reader):

    def __init__(
            self,
            name: str,
            root: Optional[str] = None,
            split: str = 'train',
            class_map: dict = None,
            input_key: str = 'image',
            target_key: str = 'label',
            target_format: Optional[str] = None,
            additional_features: Optional[List[str]] = None,
            download: bool = False,
            trust_remote_code: bool = False
    ):
        """
        """
        super().__init__()
        self.root = root
        self.split = split
        self.dataset = datasets.load_dataset(
            name,  # 'name' maps to path arg in hf datasets
            split=split,
            cache_dir=self.root,  # timm doesn't expect hidden cache dir for datasets, specify a path if root set
            **remote_code_kwargs(trust_remote_code),
        )
        # leave decode for caller, plus we want easy access to original path names...
        self.dataset = self.dataset.cast_column(input_key, datasets.Image(decode=False))

        self.image_key = input_key
        self.label_key = target_key
        self.target_keys = parse_target_keys(target_key)  # comma separated keys select binary fields
        self.dense_target = check_target_format(target_format, self.target_keys) == 'multihot'
        if self.target_keys:
            source_classes = multi_field_class_to_idx(self.target_keys, self.dataset.info.features)
        else:
            source_classes = get_class_labels(self.dataset.info, self.label_key)
        self._source_names = {index: name for name, index in source_classes.items()}
        self.remap_class = False
        if class_map:
            self.class_to_idx = load_class_map(class_map)
            self.remap_class = True
        else:
            self.class_to_idx = source_classes
        self.split_info = self.dataset.info.splits[split]
        self.num_samples = self.split_info.num_examples

        if additional_features is not None:
            if isinstance(additional_features, list):
                self.additional_features = additional_features
            else:
                self.additional_features = [additional_features]
        else:
            self.additional_features = None

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item[self.image_key]

        if 'bytes' in image and image['bytes']:
            image = io.BytesIO(image['bytes'])
        else:
            assert 'path' in image and image['path']
            image = open(image['path'], 'rb')

        label = multi_field_target(item, self.target_keys) if self.target_keys else get_field(item, self.label_key)
        if self.remap_class:
            label = remap_target(label, self.class_to_idx, self._source_names, dense=self.dense_target)

        if self.additional_features is not None:
            features = [item[feat] for feat in self.additional_features]
            return image, label, *features
        else:
            return image, label

    def __len__(self):
        return len(self.dataset)

    def _filename(self, index, basename=False, absolute=False):
        item = self.dataset[index]
        return item[self.image_key]['path']
