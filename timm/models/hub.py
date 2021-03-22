import json
import logging
import os
from functools import partial
from typing import Union, Optional

import torch
from torch.hub import load_state_dict_from_url, download_url_to_file, urlparse, HASH_REGEX
try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

from timm import __version__
try:
    from huggingface_hub import hf_hub_url
    from huggingface_hub import cached_download
    cached_download = partial(cached_download, library_name="timm", library_version=__version__)
except ImportError:
    hf_hub_url = None
    cached_download = None

_logger = logging.getLogger(__name__)


def get_cache_dir(child_dir=''):
    """
    Returns the location of the directory where models are cached (and creates it if necessary).
    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        _logger.warning('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    hub_dir = get_dir()
    child_dir = () if not child_dir else (child_dir,)
    model_dir = os.path.join(hub_dir, 'checkpoints', *child_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def download_cached_file(url, check_hash=True, progress=False):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(get_cache_dir(), filename)
    if not os.path.exists(cached_file):
        _logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def has_hf_hub(necessary=False):
    if hf_hub_url is None and necessary:
        # if no HF Hub module installed and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return hf_hub_url is not None


def hf_split(hf_id):
    rev_split = hf_id.split('@')
    assert 0 < len(rev_split) <= 2, 'hf_hub id should only contain one @ character to identify revision.'
    hf_model_id = rev_split[0]
    hf_revision = rev_split[-1] if len(rev_split) > 1 else None
    return hf_model_id, hf_revision


def load_cfg_from_json(json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)


def _download_from_hf(model_id: str, filename: str):
    hf_model_id, hf_revision = hf_split(model_id)
    url = hf_hub_url(hf_model_id, filename, revision=hf_revision)
    return cached_download(url, cache_dir=get_cache_dir('hf'))


def load_model_config_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, 'config.json')
    default_cfg = load_cfg_from_json(cached_file)
    default_cfg['hf_hub'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    model_name = default_cfg.get('architecture')
    return default_cfg, model_name


def load_state_dict_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, 'pytorch_model.bin')
    state_dict = torch.load(cached_file, map_location='cpu')
    return state_dict
