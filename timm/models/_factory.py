import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from torch import nn

from timm.layers import set_layer_config
from ._helpers import load_checkpoint
from ._hub import load_model_config_from_hf, load_model_config_from_path
from ._pretrained import PretrainedCfg
from ._registry import is_model, model_entrypoint, split_model_name_tag


__all__ = ['parse_model_name', 'safe_model_name', 'create_model']

# Model sources that may be specified as a URI-like scheme prefix on model names.
_MODEL_SOURCES = ('hf-hub', 'local-dir')
# Deprecated source aliases, mapped to their current equivalent.
_MODEL_SOURCE_ALIASES = {'hf_hub': 'hf-hub'}
# A source prefix, requires 2+ chars so single letter Windows drive specifiers (C:\...) are not matched.
_SOURCE_PREFIX_PATTERN = re.compile(r'[a-zA-Z][a-zA-Z0-9+.\-_]+')
# Characters that never appear in a timm registry model name, both platforms' separators + drive/scheme colon.
_PATH_CHARS = ('/', '\\', ':')


def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    """Parse source and name from potentially prefixed model name.

    Everything after a recognized source prefix is treated as an opaque identifier (a Hugging Face
    Hub repo id or a local filesystem path). It is deliberately not URL parsed, so Windows drive
    letters, UNC paths, and paths containing characters such as '#', '?' or '@' pass through as-is.

    Args:
        model_name: Model name, optionally prefixed with a source, e.g. ``hf-hub:timm/resnet18.a1_in1k``
            or ``local-dir:/path/to/model_folder``.

    Returns:
        Tuple of (source, model_id). Source is None for plain (timm registry) model names.

    Raises:
        ValueError: If a source prefix is present but unknown, or has an empty model id. Also if no prefix
            is present but the name looks like a path or Hub repo id, as falling back to a registry model
            of the same basename would silently load different weights (or drop the repo owner).
    """
    source, sep, model_id = model_name.partition(':')
    if not sep or not _SOURCE_PREFIX_PATTERN.fullmatch(source):
        # no source prefix, must name a model in the timm registry
        if any(c in model_name for c in _PATH_CHARS):
            # a Hub repo id and a relative folder are not distinguishable here, so name both sources
            raise ValueError(
                f"Model name '{model_name}' has no source prefix but looks like a Hugging Face Hub repo id"
                f" or a local path. Use 'hf-hub:{model_name}' to load from the Hub"
                f" or 'local-dir:{model_name}' to load from a local folder."
            )
        return None, model_name

    source = source.lower()
    # NOTE for backwards compat, deprecated hf_hub use
    source = _MODEL_SOURCE_ALIASES.get(source, source)
    if source not in _MODEL_SOURCES:
        raise ValueError(
            f"Unknown model source '{source}' in model name '{model_name}',"
            f" valid sources: {', '.join(_MODEL_SOURCES)}."
        )
    if not model_id:
        raise ValueError(f"Model name '{model_name}' has no model id / path after the '{source}:' prefix.")

    # FIXME may use fragment as revision, currently `@` in URI path
    return source, model_id


def safe_model_name(model_name: str, remove_source: bool = True) -> str:
    """Return a filename / path safe model name."""
    def make_safe(name: str) -> str:
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')
    if remove_source:
        model_name = parse_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(
        model_name: str,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
        pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        scriptable: Optional[bool] = None,
        exportable: Optional[bool] = None,
        no_jit: Optional[bool] = None,
        **kwargs: Any,
) -> nn.Module:
    """Create a model.

    Lookup model's entrypoint function and pass relevant args to create a new model.

    Tip:
        **kwargs will be passed through entrypoint fn to ``timm.models.build_model_with_cfg()``
        and then the model class __init__(). kwargs values set to None are pruned before passing.

    Args:
        model_name: Name of model to instantiate.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        pretrained_cfg: Pass in an external pretrained_cfg for model.
        pretrained_cfg_overlay: Replace key-values in base pretrained_cfg with these.
        checkpoint_path: Path of checkpoint to load _after_ the model is initialized.
        cache_dir: Override model cache dir for Hugging Face Hub and Torch checkpoints.
        scriptable: Set layer config so that model is jit scriptable (not working for all models yet).
        exportable: Set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet).
        no_jit: Set layer config so that model doesn't utilize jit scripted layers (so far activations only).

    Keyword Args:
        drop_rate (float): Classifier dropout rate for training.
        drop_path_rate (float): Stochastic depth drop rate for training.
        global_pool (str): Classifier global pooling type.

    Example:

    ```py
    >>> from timm import create_model

    >>> # Create a MobileNetV3-Large model with no pretrained weights.
    >>> model = create_model('mobilenetv3_large_100')

    >>> # Create a MobileNetV3-Large model with pretrained weights.
    >>> model = create_model('mobilenetv3_large_100', pretrained=True)
    >>> model.num_classes
    1000

    >>> # Create a MobileNetV3-Large model with pretrained weights and a new head with 10 classes.
    >>> model = create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
    >>> model.num_classes
    10

    >>> # Create a Dinov2 small model with pretrained weights and save weights in a custom directory.
    >>> model = create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, cache_dir="/data/my-models")
    >>> # Data will be stored at `/data/my-models/models--timm--vit_small_patch14_dinov2.lvd142m/`
    ```
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_id = parse_model_name(model_name)
    if model_source:
        assert not pretrained_cfg, 'pretrained_cfg should not be set when sourcing model from Hugging Face Hub.'
        if model_source == 'hf-hub':
            # For model names specified in the form `hf-hub:path/architecture_name@revision`,
            # load model weights + pretrained_cfg from Hugging Face hub.
            pretrained_cfg, model_name, model_args = load_model_config_from_hf(
                model_id,
                cache_dir=cache_dir,
            )
        elif model_source == 'local-dir':
            pretrained_cfg, model_name, model_args = load_model_config_from_path(
                model_id,
            )
        else:
            assert False, f'Unknown model_source {model_source}'
        if model_args:
            for k, v in model_args.items():
                kwargs.setdefault(k, v)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_id)
        if pretrained_tag and not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            cache_dir=cache_dir,
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
