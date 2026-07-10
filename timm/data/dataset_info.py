from abc import ABC, abstractmethod
from numbers import Integral
from typing import Dict, List, NamedTuple, Optional, Tuple, Union


LabelNames = Union[List[str], Tuple[str, ...], Dict[Union[int, str], str]]


class DatasetInfo(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def num_classes(self):
        pass

    @abstractmethod
    def label_names(self):
        pass

    def label_indices(self) -> Tuple[int, ...]:
        """Return classifier indices that have label names."""
        return tuple(range(self.num_classes()))

    def has_label(self, index) -> bool:
        """Return whether a classifier index has a label name."""
        return 0 <= int(index) < self.num_classes()

    @abstractmethod
    def label_descriptions(self, detailed: bool = False, as_dict: bool = False) -> Union[List[str], Dict[str, str]]:
        pass

    @abstractmethod
    def index_to_label_name(self, index) -> str:
        pass

    @abstractmethod
    def index_to_description(self, index: int, detailed: bool = False) -> str:
        pass

    @abstractmethod
    def label_name_to_description(self, label: str, detailed: bool = False) -> str:
        pass


class CustomDatasetInfo(DatasetInfo):
    """ DatasetInfo that wraps passed values for custom datasets."""

    def __init__(
            self,
            label_names: LabelNames,
            label_descriptions: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        if not isinstance(label_names, (list, tuple, dict)) or not label_names:
            raise ValueError('label_names must be a non-empty list, tuple, or index-to-name dictionary.')

        # Preserve the caller-facing container for backwards compatibility.
        self._label_names = label_names
        self._label_names_by_index = None
        if isinstance(label_names, dict):
            # JSON object keys are always strings, so normalize index keys when
            # loading model configs from the Hub. Dicts may be sparse.
            normalized_label_names = {}
            for raw_index, label_name in label_names.items():
                if not isinstance(raw_index, (Integral, str)):
                    raise TypeError(f'Label index must be an int or string, got {type(raw_index).__name__}.')
                try:
                    index = int(raw_index)
                except ValueError as e:
                    raise ValueError(f'Label index must be integer-like, got {raw_index!r}.') from e
                if index < 0:
                    raise ValueError(f'Label index must be non-negative, got {index}.')
                if index in normalized_label_names:
                    raise ValueError(f'Duplicate label index after normalization: {index}.')
                normalized_label_names[index] = label_name
            self._label_names_by_index = normalized_label_names
            label_name_values = normalized_label_names.values()
        else:
            label_name_values = label_names

        if not all(isinstance(name, str) for name in label_name_values):
            raise TypeError('All label names must be strings.')

        self._label_descriptions = label_descriptions  # label name => label description mapping
        if self._label_descriptions is not None:
            # validate descriptions (label names required)
            if not isinstance(self._label_descriptions, dict):
                raise TypeError('label_descriptions must be a label-name-to-description dictionary.')
            missing_descriptions = [name for name in label_name_values if name not in self._label_descriptions]
            if missing_descriptions:
                raise ValueError(f'Missing descriptions for label names: {missing_descriptions}.')

    def num_classes(self):
        return len(self._label_names)

    def label_names(self):
        return self._label_names

    def label_indices(self) -> Tuple[int, ...]:
        """Return classifier indices that have label names."""
        if self._label_names_by_index is not None:
            return tuple(self._label_names_by_index)
        return tuple(range(len(self._label_names)))

    def has_label(self, index) -> bool:
        """Return whether a classifier index has a label name."""
        index = int(index)
        if self._label_names_by_index is not None:
            return index in self._label_names_by_index
        return 0 <= index < len(self._label_names)

    def label_descriptions(self, detailed: bool = False, as_dict: bool = False) -> Union[List[str], Dict[str, str]]:
        return self._label_descriptions

    def label_name_to_description(self, label: str, detailed: bool = False) -> str:
        if self._label_descriptions:
            return self._label_descriptions[label]
        return label  # return label name itself if a descriptions is not present

    def index_to_label_name(self, index) -> str:
        if self._label_names_by_index is not None:
            return self._label_names_by_index[int(index)]
        assert 0 <= index < len(self._label_names)
        return self._label_names[index]

    def index_to_description(self, index: int, detailed: bool = False) -> str:
        label = self.index_to_label_name(index)
        return self.label_name_to_description(label, detailed=detailed)


class LabelMappingCoverage(NamedTuple):
    mapped: int
    missing: int
    extra: int


class DatasetInfoLabelMapper:
    """Map classifier indices to labels with reusable sparse-mapping fallback behavior."""

    def __init__(
            self,
            dataset_info: DatasetInfo,
            label_type: str = 'description',
            fallback_format: Optional[str] = '<unmapped:{index}>',
    ):
        if label_type not in ('name', 'description', 'detail', 'detailed'):
            raise ValueError(f'Invalid label type: {label_type}.')
        self.dataset_info = dataset_info
        self.label_type = label_type
        self.fallback_format = fallback_format

    def __call__(self, index) -> str:
        index = int(index)
        if not self.dataset_info.has_label(index) and self.fallback_format is not None:
            return self.fallback_format.format(index=index)
        if self.label_type == 'name':
            return self.dataset_info.index_to_label_name(index)
        return self.dataset_info.index_to_description(
            index,
            detailed=self.label_type in ('detail', 'detailed'),
        )

    def coverage(self, num_classes: int) -> LabelMappingCoverage:
        """Return mapped, missing, and out-of-range label counts for a classifier."""
        if num_classes < 0:
            raise ValueError(f'num_classes must be non-negative, got {num_classes}.')
        mapped_indices = self.dataset_info.label_indices()
        mapped = sum(0 <= index < num_classes for index in mapped_indices)
        return LabelMappingCoverage(
            mapped=mapped,
            missing=num_classes - mapped,
            extra=len(mapped_indices) - mapped,
        )
