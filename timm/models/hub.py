import json
import logging
import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import torch
from torch.hub import HASH_REGEX, download_url_to_file, urlparse

try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

from timm import __version__
from timm.models.pretrained import filter_pretrained_cfg

try:
    from huggingface_hub import (
        create_repo, get_hf_file_metadata,
        hf_hub_download, hf_hub_url,
        repo_type_and_id_from_hf_id, upload_folder)
    from huggingface_hub.utils import EntryNotFoundError
    hf_hub_download = partial(hf_hub_download, library_name="timm", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False

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
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
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
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def hf_split(hf_id):
    # FIXME I may change @ -> # and be parsed as fragment in a URI model name scheme
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
    return hf_hub_download(hf_model_id, filename, revision=hf_revision)


def load_model_config_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, 'config.json')

    hf_config = load_cfg_from_json(cached_file)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:
            hf_config['label_name'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg

    # NOTE currently discarding parent config as only arch name and pretrained_cfg used in timm right now
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'
    if 'num_classes' in hf_config:
        # model should be created with parent num_classes if they exist
        pretrained_cfg['num_classes'] = hf_config['num_classes']
    model_name = hf_config['architecture']

    return pretrained_cfg, model_name


def load_state_dict_from_hf(model_id: str, filename: str = 'pytorch_model.bin'):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, filename)
    state_dict = torch.load(cached_file, map_location='cpu')
    return state_dict


def save_for_hf(model, save_directory, model_config=None):
    assert has_hf_hub(True)
    model_config = model_config or {}
    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True, parents=True)

    weights_path = save_directory / 'pytorch_model.bin'
    torch.save(model.state_dict(), weights_path)

    config_path = save_directory / 'config.json'
    hf_config = {}
    pretrained_cfg = filter_pretrained_cfg(model.pretrained_cfg, remove_source=True, remove_null=True)
    # set some values at root config level
    hf_config['architecture'] = pretrained_cfg.pop('architecture')
    hf_config['num_classes'] = model_config.get('num_classes', model.num_classes)
    hf_config['num_features'] = model_config.get('num_features', model.num_features)
    hf_config['global_pool'] = model_config.get('global_pool', getattr(model, 'global_pool', None))

    if 'label' in model_config:
        _logger.warning(
            "'label' as a config field for timm models is deprecated. Please use 'label_name' and 'display_name'. "
            "Using provided 'label' field as 'label_name'.")
        model_config['label_name'] = model_config.pop('label')

    label_name = model_config.pop('label_name', None)
    if label_name:
        assert isinstance(label_name, (dict, list, tuple))
        # map label id (classifier index) -> unique label name (ie synset for ImageNet, MID for OpenImages)
        # can be a dict id: name if there are id gaps, or tuple/list if no gaps.
        hf_config['label_name'] = model_config['label_name']

    display_name = model_config.pop('display_name', None)
    if display_name:
        assert isinstance(display_name, dict)
        # map label_name -> user interface display name
        hf_config['display_name'] = model_config['display_name']

    hf_config['pretrained_cfg'] = pretrained_cfg
    hf_config.update(model_config)

    with config_path.open('w') as f:
        json.dump(hf_config, f, indent=2)


def push_to_hf_hub(
    model,
    repo_id: str,
    commit_message: str = 'Add model',
    token: Optional[str] = None,
    revision: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
    model_config: Optional[dict] = None,
):
    # Create repo if it doesn't exist yet
    repo_url = create_repo(repo_id, token=token, private=private, exist_ok=True)

    # Infer complete repo_id from repo_url
    # Can be different from the input `repo_id` if repo_owner was implicit
    _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
    repo_id = f"{repo_owner}/{repo_name}"

    # Check if README file already exist in repo
    try:
        get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
        has_readme = True
    except EntryNotFoundError:
        has_readme = False

    # Dump model and push to Hub
    with TemporaryDirectory() as tmpdir:
        # Save model weights and config.
        save_for_hf(model, tmpdir, model_config=model_config)

        # Add readme if it does not exist
        if not has_readme:
            model_name = repo_id.split('/')[-1]
            readme_path = Path(tmpdir) / "README.md"
            readme_text = f'---\ntags:\n- image-classification\n- timm\nlibrary_tag: timm\n---\n# Model card for {model_name}'
            readme_path.write_text(readme_text)

        # Upload model and return
        return upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            revision=revision,
            create_pr=create_pr,
            commit_message=commit_message,
        )
