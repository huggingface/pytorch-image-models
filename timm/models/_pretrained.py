import copy
from collections import deque, defaultdict
from dataclasses import dataclass, field, replace, asdict
from typing import Any, Deque, Dict, Tuple, Optional, Union


__all__ = ['PretrainedCfg', 'filter_pretrained_cfg', 'DefaultCfg']


@dataclass
class PretrainedCfg:
    """
    """
    # weight source locations
    url: Optional[Union[str, Tuple[str, str]]] = None  # remote URL
    file: Optional[str] = None  # local / shared filesystem path
    state_dict: Optional[Dict[str, Any]] = None  # in-memory state dict
    hf_hub_id: Optional[str] = None  # Hugging Face Hub model id ('organization/model')
    hf_hub_filename: Optional[str] = None  # Hugging Face Hub filename (overrides default)

    source: Optional[str] = None  # source of cfg / weight location used (url, file, hf-hub)
    architecture: Optional[str] = None  # architecture variant can be set when not implicit
    tag: Optional[str] = None  # pretrained tag of source
    custom_load: bool = False  # use custom model specific model.load_pretrained() (ie for npz files)

    # input / data config
    input_size: Tuple[int, int, int] = (3, 224, 224)
    test_input_size: Optional[Tuple[int, int, int]] = None
    min_input_size: Optional[Tuple[int, int, int]] = None
    fixed_input_size: bool = False
    interpolation: str = 'bicubic'
    crop_pct: float = 0.875
    test_crop_pct: Optional[float] = None
    crop_mode: str = 'center'
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # head / classifier config and meta-data
    num_classes: int = 1000
    label_offset: Optional[int] = None
    label_names: Optional[Tuple[str]] = None
    label_descriptions: Optional[Dict[str, str]] = None

    # model attributes that vary with above or required for pretrained adaptation
    pool_size: Optional[Tuple[int, ...]] = None
    test_pool_size: Optional[Tuple[int, ...]] = None
    first_conv: Optional[str] = None
    classifier: Optional[str] = None

    license: Optional[str] = None
    description: Optional[str] = None
    origin_url: Optional[str] = None
    paper_name: Optional[str] = None
    paper_ids: Optional[Union[str, Tuple[str]]] = None
    notes: Optional[Tuple[str]] = None

    @property
    def has_weights(self):
        return self.url or self.file or self.hf_hub_id

    def to_dict(self, remove_source=False, remove_null=True):
        return filter_pretrained_cfg(
            asdict(self),
            remove_source=remove_source,
            remove_null=remove_null
        )


def filter_pretrained_cfg(cfg, remove_source=False, remove_null=True):
    filtered_cfg = {}
    keep_null = {'pool_size', 'first_conv', 'classifier'}  # always keep these keys, even if none
    for k, v in cfg.items():
        if remove_source and k in {'url', 'file', 'hf_hub_id', 'hf_hub_id', 'hf_hub_filename', 'source'}:
            continue
        if remove_null and v is None and k not in keep_null:
            continue
        filtered_cfg[k] = v
    return filtered_cfg


@dataclass
class DefaultCfg:
    tags: Deque[str] = field(default_factory=deque)  # priority queue of tags (first is default)
    cfgs: Dict[str, PretrainedCfg] = field(default_factory=dict)  # pretrained cfgs by tag
    is_pretrained: bool = False  # at least one of the configs has a pretrained source set

    @property
    def default(self):
        return self.cfgs[self.tags[0]]

    @property
    def default_with_tag(self):
        tag = self.tags[0]
        return tag, self.cfgs[tag]
