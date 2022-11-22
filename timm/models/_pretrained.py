from collections import deque, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Deque, Dict, Tuple, Optional, Union


@dataclass
class PretrainedCfg:
    """
    """
    # weight locations
    url: str = ''
    file: str = ''
    hf_hub_id: str = ''
    hf_hub_filename: str = ''

    source: str = ''  # source of cfg / weight location used (url, file, hf-hub)
    architecture: str = ''  # architecture variant can be set when not implicit
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
    label_offset: int = 0

    # model attributes that vary with above or required for pretrained adaptation
    pool_size: Optional[Tuple[int, ...]] = None
    test_pool_size: Optional[Tuple[int, ...]] = None
    first_conv: str = ''
    classifier: str = ''

    license: str = ''
    source_url: str = ''
    paper: str = ''
    notes: str = ''

    @property
    def has_weights(self):
        return self.url.startswith('http') or self.file or self.hf_hub_id


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


def generate_defaults(cfgs: Dict[str, Union[Dict[str, Any], PretrainedCfg]]):
    out = defaultdict(DefaultCfg)
    default_set = set()  # no tag and tags ending with * are prioritized as default

    for k, v in cfgs.items():
        if isinstance(v, dict):
            v = PretrainedCfg(**v)
        has_weights = v.has_weights

        model, tag = split_model_name_tag(k)
        is_default_set = model in default_set
        priority = not tag or (tag.endswith('*') and not is_default_set)
        tag = tag.strip('*')

        default_cfg = out[model]
        if has_weights:
            default_cfg.is_pretrained = True

        if priority:
            default_cfg.tags.appendleft(tag)
            default_set.add(model)
        elif has_weights and not default_set:
            default_cfg.tags.appendleft(tag)
        else:
            default_cfg.tags.append(tag)

        default_cfg.cfgs[tag] = v

    return out
