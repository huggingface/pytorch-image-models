import copy
from collections import deque, defaultdict
from dataclasses import dataclass, field, replace, asdict
from typing import Any, Deque, Dict, Tuple, Optional, Union


__all__ = ['PretrainedCfg', 'filter_pretrained_cfg', 'DefaultCfg', 'split_model_name_tag', 'generate_default_cfgs']


@dataclass
class PretrainedCfg:
    """
    """
    # weight locations
    url: Optional[Union[str, Tuple[str, str]]] = None
    file: Optional[str] = None
    hf_hub_id: Optional[str] = None
    hf_hub_filename: Optional[str] = None

    source: Optional[str] = None  # source of cfg / weight location used (url, file, hf-hub)
    architecture: Optional[str] = None  # architecture variant can be set when not implicit
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

    # head config
    num_classes: int = 1000
    label_offset: Optional[int] = None

    # model attributes that vary with above or required for pretrained adaptation
    pool_size: Optional[Tuple[int, ...]] = None
    test_pool_size: Optional[Tuple[int, ...]] = None
    first_conv: Optional[str] = None
    classifier: Optional[str] = None

    license: Optional[str] = None
    source_url: Optional[str] = None
    paper: Optional[str] = None
    notes: Optional[str] = None

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
    keep_none = {'pool_size', 'first_conv', 'classifier'}  # always keep these keys, even if none
    for k, v in cfg.items():
        if remove_source and k in {'url', 'file', 'hf_hub_id', 'hf_hub_id', 'hf_hub_filename', 'source'}:
            continue
        if remove_null and v is None and k not in keep_none:
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


def split_model_name_tag(model_name: str, no_tag=''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def generate_default_cfgs(cfgs: Dict[str, Union[Dict[str, Any], PretrainedCfg]]):
    out = defaultdict(DefaultCfg)
    default_set = set()  # no tag and tags ending with * are prioritized as default

    for k, v in cfgs.items():
        if isinstance(v, dict):
            v = PretrainedCfg(**v)
        has_weights = v.has_weights

        model, tag = split_model_name_tag(k)
        is_default_set = model in default_set
        priority = (has_weights and not tag) or (tag.endswith('*') and not is_default_set)
        tag = tag.strip('*')

        default_cfg = out[model]

        if priority:
            default_cfg.tags.appendleft(tag)
            default_set.add(model)
        elif has_weights and not default_cfg.is_pretrained:
            default_cfg.tags.appendleft(tag)
        else:
            default_cfg.tags.append(tag)

        if has_weights:
            default_cfg.is_pretrained = True

        default_cfg.cfgs[tag] = v

    return out
