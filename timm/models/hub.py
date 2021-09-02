import json
import logging
import os
from pathlib import Path
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
    from huggingface_hub import cached_download, hf_hub_url, HfFolder, HfApi, Repository
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


def save_pretrained_for_hf(model, save_directory, **config_kwargs):
    assert has_hf_hub(True)
    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True, parents=True)

    weights_path = save_directory / 'pytorch_model.bin'
    torch.save(model.state_dict(), weights_path)

    config_path = save_directory / 'config.json'
    config = model.default_cfg
    config.update(config_kwargs)

    with config_path.open('w') as f:
        json.dump(config, f, indent=4)


def push_to_hf_hub(
    model,
    repo_path_or_name: Optional[str] = None,
    repo_url: Optional[str] = None,
    commit_message: Optional[str] = "Add model",
    organization: Optional[str] = None,
    private: Optional[bool] = None,
    api_endpoint: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    git_user: Optional[str] = None,
    git_email: Optional[str] = None,
    config: Optional[dict] = None,
):
    """
    Upload model checkpoint and config to the 🤗 Model Hub while synchronizing a local clone of the repo in
    :obj:`repo_path_or_name`.

    Parameters:
        repo_path_or_name (:obj:`str`, `optional`):
            Can either be a repository name for your model or tokenizer in the Hub or a path to a local folder (in
            which case the repository will have the name of that local folder). If not specified, will default to
            the name given by :obj:`repo_url` and a local directory with that name will be created.
        repo_url (:obj:`str`, `optional`):
            Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
            repository will be created in your namespace (unless you specify an :obj:`organization`) with
            :obj:`repo_name`.
        commit_message (:obj:`str`, `optional`):
            Message to commit while pushing. Will default to :obj:`"add config"`, :obj:`"add tokenizer"` or
            :obj:`"add model"` depending on the type of the class.
        organization (:obj:`str`, `optional`):
            Organization in which you want to push your model or tokenizer (you must be a member of this
            organization).
        private (:obj:`bool`, `optional`):
            Whether or not the repository created should be private (requires a paying subscription).
        api_endpoint (:obj:`str`, `optional`):
            The API endpoint to use when pushing the model to the hub.
        use_auth_token (:obj:`bool` or :obj:`str`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`). Will default to
            :obj:`True` if :obj:`repo_url` is not specified.
        git_user (``str``, `optional`):
            will override the ``git config user.name`` for committing and pushing files to the hub.
        git_email (``str``, `optional`):
            will override the ``git config user.email`` for committing and pushing files to the hub.
        config (:obj:`dict`, `optional`):
            Configuration object to be saved alongside the model weights.


    Returns:
        The url of the commit of your model in the given repository.
    """
    assert has_hf_hub(True)
    if repo_path_or_name is None and repo_url is None:
        raise ValueError(
            "You need to specify a `repo_path_or_name` or a `repo_url`."
        )

    if use_auth_token is None and repo_url is None:
        token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "You must login to the Hugging Face hub on this computer by typing `transformers-cli login` and "
                "entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own "
                "token as the `use_auth_token` argument."
            )
    elif isinstance(use_auth_token, str):
        token = use_auth_token
    else:
        token = None

    if repo_path_or_name is None:
        repo_path_or_name = repo_url.split("/")[-1]

    # If no URL is passed and there's no path to a directory containing files, create a repo
    if repo_url is None and not os.path.exists(repo_path_or_name):
        repo_name = Path(repo_path_or_name).name
        repo_url = HfApi(endpoint=api_endpoint).create_repo(
            token,
            repo_name,
            organization=organization,
            private=private,
            repo_type=None,
            exist_ok=True,
        )

    repo = Repository(
        repo_path_or_name,
        clone_from=repo_url,
        use_auth_token=use_auth_token,
        git_user=git_user,
        git_email=git_email,
    )
    repo.git_pull(rebase=True)

    save_config = model.default_cfg
    save_config.update(config or {})
    with repo.commit(commit_message):
        save_pretrained_for_hf(model, repo.local_dir, **save_config)
