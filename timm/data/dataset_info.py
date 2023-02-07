from abc import ABC, abstractmethod
from typing import Dict, List, Union


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