import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch.hub import HASH_REGEX, download_url_to_file, urlparse
try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

from timm import __version__
try:
    from huggingface_hub import HfApi, HfFolder, Repository, cached_download, hf_hub_url
    cached_download = partial(cached_download, library_name="timm", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    cached_download = None
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
        # if no HF Hub module installed and it is necessary to continue, raise error
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
    url = hf_hub_url(hf_model_id, filename, revision=hf_revision)
    return cached_download(url, cache_dir=get_cache_dir('hf'))


def load_model_config_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, 'config.json')
    pretrained_cfg = load_cfg_from_json(cached_file)
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'
    model_name = pretrained_cfg.get('architecture')
    return pretrained_cfg, model_name


def load_state_dict_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, 'pytorch_model.bin')
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
    hf_config = model.pretrained_cfg
    hf_config['num_classes'] = model_config.pop('num_classes', model.num_classes)
    hf_config['num_features'] = model_config.pop('num_features', model.num_features)
    hf_config['labels'] = model_config.pop('labels', [f"LABEL_{i}" for i in range(hf_config['num_classes'])])
    hf_config.update(model_config)

    with config_path.open('w') as f:
        json.dump(hf_config, f, indent=2)


def push_to_hf_hub(
    model,
    local_dir,
    repo_namespace_or_url=None,
    commit_message='Add model',
    use_auth_token=True,
    git_email=None,
    git_user=None,
    revision=None,
    model_config=None,
):
    if repo_namespace_or_url:
        repo_owner, repo_name = repo_namespace_or_url.rstrip('/').split('/')[-2:]
    else:
        if isinstance(use_auth_token, str):
            token = use_auth_token
        else:
            token = HfFolder.get_token()

        if token is None:
            raise ValueError(
                "You must login to the Hugging Face hub on this computer by typing `transformers-cli login` and "
                "entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own "
                "token as the `use_auth_token` argument."
            )

        repo_owner = HfApi().whoami(token)['name']
        repo_name = Path(local_dir).name

    repo_url = f'https://huggingface.co/{repo_owner}/{repo_name}'

    repo = Repository(
        local_dir,
        clone_from=repo_url,
        use_auth_token=use_auth_token,
        git_user=git_user,
        git_email=git_email,
        revision=revision,
    )

    # Prepare a default model card that includes the necessary tags to enable inference.
    readme_text = f'---\ntags:\n- image-classification\n- timm\nlibrary_tag: timm\n---\n# Model card for {repo_name}'
    with repo.commit(commit_message):
        # Save model weights and config.
        save_for_hf(model, repo.local_dir, model_config=model_config)

        # Save a model card if it doesn't exist.
        readme_path = Path(repo.local_dir) / 'README.md'
        if not readme_path.exists():
            readme_path.write_text(readme_text)

    return repo.git_remote_url()
