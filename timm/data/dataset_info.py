from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class DatasetInfo(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def num_classes(self):
        pass

    @abstractmethod
    def label_names(self):
        pass

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
            label_names: Union[List[str], Dict[int, str]],
            label_descriptions: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        assert len(label_names) > 0
        self._label_names = label_names  # label index => label name mapping
        self._label_descriptions = label_descriptions  # label name => label description mapping
        if self._label_descriptions is not None:
            # validate descriptions (label names required)
            assert isinstance(self._label_descriptions, dict)
            for n in self._label_names:
                assert n in self._label_descriptions

    def num_classes(self):
        return len(self._label_names)

    def label_names(self):
        return self._label_names

    def label_descriptions(self, detailed: bool = False, as_dict: bool = False) -> Union[List[str], Dict[str, str]]:
        return self._label_descriptions

    def label_name_to_description(self, label: str, detailed: bool = False) -> str:
        if self._label_descriptions:
            return self._label_descriptions[label]
        return label  # return label name itself if a descriptions is not present

    def index_to_label_name(self, index) -> str:
        assert 0 <= index < len(self._label_names)
        return self._label_names[index]

    def index_to_description(self, index: int, detailed: bool = False) -> str:
        label = self.index_to_label_name(index)
        return self.label_name_to_description(label, detailed=detailed)
