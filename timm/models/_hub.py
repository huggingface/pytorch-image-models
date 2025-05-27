import hashlib
import json
import logging
import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.hub import HASH_REGEX, download_url_to_file, urlparse

try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from timm import __version__
from timm.models._pretrained import filter_pretrained_cfg

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

__all__ = ['get_cache_dir', 'download_cached_file', 'has_hf_hub', 'hf_split', 'load_model_config_from_hf',
           'load_state_dict_from_hf', 'save_for_hf', 'push_to_hf_hub']

# Default name for a weights file hosted on the Huggingface Hub.
HF_WEIGHTS_NAME = "pytorch_model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "model.safetensors"  # safetensors version
HF_OPEN_CLIP_WEIGHTS_NAME = "open_clip_pytorch_model.bin"  # default pytorch pkl
HF_OPEN_CLIP_SAFE_WEIGHTS_NAME = "open_clip_model.safetensors"  # safetensors version


def get_cache_dir(child_dir: str = ''):
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


def download_cached_file(
        url: Union[str, List[str], Tuple[str, str]],
        check_hash: bool = True,
        progress: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
):
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = get_cache_dir()
    cached_file = os.path.join(cache_dir, filename)
    if not os.path.exists(cached_file):
        _logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def check_cached_file(
        url: Union[str, List[str], Tuple[str, str]],
        check_hash: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
):
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    if not cache_dir:
        cache_dir = get_cache_dir()
    cached_file = os.path.join(cache_dir, filename)
    if os.path.exists(cached_file):
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
            if hash_prefix:
                with open(cached_file, 'rb') as f:
                    hd = hashlib.sha256(f.read()).hexdigest()
                    if hd[:len(hash_prefix)] != hash_prefix:
                        return False
        return True
    return False


def has_hf_hub(necessary: bool = False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def hf_split(hf_id: str):
    # FIXME I may change @ -> # and be parsed as fragment in a URI model name scheme
    rev_split = hf_id.split('@')
    assert 0 < len(rev_split) <= 2, 'hf_hub id should only contain one @ character to identify revision.'
    hf_model_id = rev_split[0]
    hf_revision = rev_split[-1] if len(rev_split) > 1 else None
    return hf_model_id, hf_revision


def load_cfg_from_json(json_file: Union[str, Path]):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)


def download_from_hf(
        model_id: str,
        filename: str,
        cache_dir: Optional[Union[str, Path]] = None,
):
    hf_model_id, hf_revision = hf_split(model_id)
    return hf_hub_download(
        hf_model_id,
        filename,
        revision=hf_revision,
        cache_dir=cache_dir,
    )


def _parse_model_cfg(
        cfg: Dict[str, Any],
        extra_fields: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """"""
    # legacy "single‑dict" → split
    if "pretrained_cfg" not in cfg:
        pretrained_cfg = cfg
        cfg = {
            "architecture": pretrained_cfg.pop("architecture"),
            "num_features": pretrained_cfg.pop("num_features", None),
            "pretrained_cfg": pretrained_cfg,
        }
        if "labels" in pretrained_cfg:                  # rename ‑‑> label_names
            pretrained_cfg["label_names"] = pretrained_cfg.pop("labels")

    pretrained_cfg = cfg["pretrained_cfg"]
    pretrained_cfg.update(extra_fields)

    # top‑level overrides
    if "num_classes" in cfg:
        pretrained_cfg["num_classes"] = cfg["num_classes"]
    if "label_names" in cfg:
        pretrained_cfg["label_names"] = cfg.pop("label_names")
    if "label_descriptions" in cfg:
        pretrained_cfg["label_descriptions"] = cfg.pop("label_descriptions")

    model_args = cfg.get("model_args", {})
    model_name = cfg["architecture"]
    return pretrained_cfg, model_name, model_args


def load_model_config_from_hf(
        model_id: str,
        cache_dir: Optional[Union[str, Path]] = None,
):
    """Original HF‑Hub loader (unchanged download, shared parsing)."""
    assert has_hf_hub(True)
    cfg_path = download_from_hf(model_id, "config.json", cache_dir=cache_dir)
    cfg = load_cfg_from_json(cfg_path)
    return _parse_model_cfg(cfg, {"hf_hub_id": model_id, "source": "hf-hub"})


def load_model_config_from_path(
        model_path: Union[str, Path],
):
    """Load from ``<model_path>/config.json`` on the local filesystem."""
    model_path = Path(model_path)
    cfg_file = model_path / "config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    cfg = load_cfg_from_json(cfg_file)
    extra_fields = {"file": str(model_path), "source": "local-dir"}
    return _parse_model_cfg(cfg, extra_fields=extra_fields)


def load_state_dict_from_hf(
        model_id: str,
        filename: str = HF_WEIGHTS_NAME,
        weights_only: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
):
    assert has_hf_hub(True)
    hf_model_id, hf_revision = hf_split(model_id)

    # Look for .safetensors alternatives and load from it if it exists
    if _has_safetensors:
        for safe_filename in _get_safe_alternatives(filename):
            try:
                cached_safe_file = hf_hub_download(
                    repo_id=hf_model_id,
                    filename=safe_filename,
                    revision=hf_revision,
                    cache_dir=cache_dir,
                )
                _logger.info(
                    f"[{model_id}] Safe alternative available for '{filename}' "
                    f"(as '{safe_filename}'). Loading weights using safetensors.")
                return safetensors.torch.load_file(cached_safe_file, device="cpu")
            except EntryNotFoundError:
                pass

    # Otherwise, load using pytorch.load
    cached_file = hf_hub_download(
        hf_model_id,
        filename=filename,
        revision=hf_revision,
        cache_dir=cache_dir,
    )
    _logger.debug(f"[{model_id}] Safe alternative not found for '{filename}'. Loading weights using default pytorch.")
    try:
        state_dict = torch.load(cached_file, map_location='cpu', weights_only=weights_only)
    except TypeError:
        state_dict = torch.load(cached_file, map_location='cpu')
    return state_dict


_PREFERRED_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.pth",
    "model.pth",
    "open_clip_model.safetensors",
    "open_clip_pytorch_model.safetensors",
    "open_clip_pytorch_model.bin",
    "open_clip_pytorch_model.pth",
)
_EXT_PRIORITY = ('.safetensors', '.pth', '.pth.tar', '.bin')

def load_state_dict_from_path(
        path: str,
        weights_only: bool = False,
):
    found_file = None
    for fname in _PREFERRED_FILES:
        p = path / fname
        if p.exists():
            logging.info(f"Found preferred checkpoint: {p.name}")
            found_file = p
            break

    # fallback: first match per‑extension class
    for ext in _EXT_PRIORITY:
        files = sorted(path.glob(f"*{ext}"))
        if files:
            if len(files) > 1:
                logging.warning(
                    f"Multiple {ext} checkpoints in {path}: {names}. "
                    f"Using '{files[0].name}'."
                )
            found_file = files[0]

    if not found_file:
        raise RuntimeError(f"No suitable checkpoints found in {path}.")

    try:
        state_dict = torch.load(found_file, map_location='cpu', weights_only=weights_only)
    except TypeError:
        state_dict = torch.load(found_file, map_location='cpu')
    return state_dict


def load_custom_from_hf(
        model_id: str,
        filename: str,
        model: torch.nn.Module,
        cache_dir: Optional[Union[str, Path]] = None,
):
    assert has_hf_hub(True)
    hf_model_id, hf_revision = hf_split(model_id)
    cached_file = hf_hub_download(
        hf_model_id,
        filename=filename,
        revision=hf_revision,
        cache_dir=cache_dir,
    )
    return model.load_pretrained(cached_file)


def save_config_for_hf(
        model: torch.nn.Module,
        config_path: str,
        model_config: Optional[dict] = None,
        model_args: Optional[dict] = None
):
    model_config = model_config or {}
    hf_config = {}
    pretrained_cfg = filter_pretrained_cfg(model.pretrained_cfg, remove_source=True, remove_null=True)
    # set some values at root config level
    hf_config['architecture'] = pretrained_cfg.pop('architecture')
    hf_config['num_classes'] = model_config.pop('num_classes', model.num_classes)

    # NOTE these attr saved for informational purposes, do not impact model build
    hf_config['num_features'] = model_config.pop('num_features', model.num_features)
    global_pool_type = model_config.pop('global_pool', getattr(model, 'global_pool', None))
    if isinstance(global_pool_type, str) and global_pool_type:
        hf_config['global_pool'] = global_pool_type

    # Save class label info
    if 'labels' in model_config:
        _logger.warning(
            "'labels' as a config field for is deprecated. Please use 'label_names' and 'label_descriptions'."
            " Renaming provided 'labels' field to 'label_names'.")
        model_config.setdefault('label_names', model_config.pop('labels'))

    label_names = model_config.pop('label_names', None)
    if label_names:
        assert isinstance(label_names, (dict, list, tuple))
        # map label id (classifier index) -> unique label name (ie synset for ImageNet, MID for OpenImages)
        # can be a dict id: name if there are id gaps, or tuple/list if no gaps.
        hf_config['label_names'] = label_names

    label_descriptions = model_config.pop('label_descriptions', None)
    if label_descriptions:
        assert isinstance(label_descriptions, dict)
        # maps label names -> descriptions
        hf_config['label_descriptions'] = label_descriptions

    if model_args:
        hf_config['model_args'] = model_args

    hf_config['pretrained_cfg'] = pretrained_cfg
    hf_config.update(model_config)

    with config_path.open('w') as f:
        json.dump(hf_config, f, indent=2)


def save_for_hf(
        model: torch.nn.Module,
        save_directory: str,
        model_config: Optional[dict] = None,
        model_args: Optional[dict] = None,
        safe_serialization: Union[bool, Literal["both"]] = False,
):
    assert has_hf_hub(True)
    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True, parents=True)

    # Save model weights, either safely (using safetensors), or using legacy pytorch approach or both.
    tensors = model.state_dict()
    if safe_serialization is True or safe_serialization == "both":
        assert _has_safetensors, "`pip install safetensors` to use .safetensors"
        safetensors.torch.save_file(tensors, save_directory / HF_SAFE_WEIGHTS_NAME)
    if safe_serialization is False or safe_serialization == "both":
        torch.save(tensors, save_directory / HF_WEIGHTS_NAME)

    config_path = save_directory / 'config.json'
    save_config_for_hf(
        model,
        config_path,
        model_config=model_config,
        model_args=model_args,
    )


def push_to_hf_hub(
        model: torch.nn.Module,
        repo_id: str,
        commit_message: str = 'Add model',
        token: Optional[str] = None,
        revision: Optional[str] = None,
        private: bool = False,
        create_pr: bool = False,
        model_config: Optional[dict] = None,
        model_card: Optional[dict] = None,
        model_args: Optional[dict] = None,
        safe_serialization: Union[bool, Literal["both"]] = 'both',
):
    """
    Arguments:
        (...)
        safe_serialization (`bool` or `"both"`, *optional*, defaults to `False`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            Can be set to `"both"` in order to push both safe and unsafe weights.
    """
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
        save_for_hf(
            model,
            tmpdir,
            model_config=model_config,
            model_args=model_args,
            safe_serialization=safe_serialization,
        )

        # Add readme if it does not exist
        if not has_readme:
            model_card = model_card or {}
            model_name = repo_id.split('/')[-1]
            readme_path = Path(tmpdir) / "README.md"
            readme_text = generate_readme(model_card, model_name)
            readme_path.write_text(readme_text)

        # Upload model and return
        return upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            revision=revision,
            create_pr=create_pr,
            commit_message=commit_message,
        )


def generate_readme(model_card: dict, model_name: str):
    tags = model_card.get('tags', None) or ['image-classification', 'timm', 'transformers']
    readme_text = "---\n"
    if tags:
        readme_text += "tags:\n"
        for t in tags:
            readme_text += f"- {t}\n"
    readme_text += f"library_name: {model_card.get('library_name', 'timm')}\n"
    readme_text += f"license: {model_card.get('license', 'apache-2.0')}\n"
    if 'license_name' in model_card:
        readme_text += f"license_name: {model_card.get('license_name')}\n"
    if 'license_link' in model_card:
        readme_text += f"license_link: {model_card.get('license_link')}\n"
    if 'details' in model_card and 'Dataset' in model_card['details']:
        readme_text += 'datasets:\n'
        if isinstance(model_card['details']['Dataset'], (tuple, list)):
            for d in model_card['details']['Dataset']:
                readme_text += f"- {d.lower()}\n"
        else:
            readme_text += f"- {model_card['details']['Dataset'].lower()}\n"
        if 'Pretrain Dataset' in model_card['details']:
            if isinstance(model_card['details']['Pretrain Dataset'], (tuple, list)):
                for d in model_card['details']['Pretrain Dataset']:
                    readme_text += f"- {d.lower()}\n"
            else:
                readme_text += f"- {model_card['details']['Pretrain Dataset'].lower()}\n"
    readme_text += "---\n"
    readme_text += f"# Model card for {model_name}\n"
    if 'description' in model_card:
        readme_text += f"\n{model_card['description']}\n"
    if 'details' in model_card:
        readme_text += f"\n## Model Details\n"
        for k, v in model_card['details'].items():
            if isinstance(v, (list, tuple)):
                readme_text += f"- **{k}:**\n"
                for vi in v:
                    readme_text += f"  - {vi}\n"
            elif isinstance(v, dict):
                readme_text += f"- **{k}:**\n"
                for ki, vi in v.items():
                    readme_text += f"  - {ki}: {vi}\n"
            else:
                readme_text += f"- **{k}:** {v}\n"
    if 'usage' in model_card:
        readme_text += f"\n## Model Usage\n"
        readme_text += model_card['usage']
        readme_text += '\n'

    if 'comparison' in model_card:
        readme_text += f"\n## Model Comparison\n"
        readme_text += model_card['comparison']
        readme_text += '\n'

    if 'citation' in model_card:
        readme_text += f"\n## Citation\n"
        if not isinstance(model_card['citation'], (list, tuple)):
            citations = [model_card['citation']]
        else:
            citations = model_card['citation']
        for c in citations:
            readme_text += f"```bibtex\n{c}\n```\n"
    return readme_text


def _get_safe_alternatives(filename: str) -> Iterable[str]:
    """Returns potential safetensors alternatives for a given filename.

    Use case:
        When downloading a model from the Huggingface Hub, we first look if a .safetensors file exists and if yes, we use it.
        Main use case is filename "pytorch_model.bin" => check for "model.safetensors" or "pytorch_model.safetensors".
    """
    if filename == HF_WEIGHTS_NAME:
        yield HF_SAFE_WEIGHTS_NAME
    if filename == HF_OPEN_CLIP_WEIGHTS_NAME:
        yield HF_OPEN_CLIP_SAFE_WEIGHTS_NAME
    if filename not in (HF_WEIGHTS_NAME, HF_OPEN_CLIP_WEIGHTS_NAME) and filename.endswith(".bin"):
        yield filename[:-4] + ".safetensors"
