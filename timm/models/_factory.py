import os
from typing import Any, Dict, Optional, Union
from urllib.parse import urlsplit

from timm.layers import set_layer_config
from ._helpers import load_checkpoint
from ._hub import load_model_config_from_hf
from ._pretrained import PretrainedCfg
from ._registry import is_model, model_entrypoint, split_model_name_tag


__all__ = ['parse_model_name', 'safe_model_name', 'create_model']


def parse_model_name(model_name: str):
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def safe_model_name(model_name: str, remove_source: bool = True):
    # return a filename / path safe model name
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')
    if remove_source:
        model_name = parse_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(
        model_name: str,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
        pretrained_cfg_overlay:  Optional[Dict[str, Any]] = None,
        checkpoint_path: str = '',
        scriptable: Optional[bool] = None,
        exportable: Optional[bool] = None,
        no_jit: Optional[bool] = None,
        **kwargs,
):
    """Create a model.

    Lookup model's entrypoint function and pass relevant args to create a new model.

    <Tip>
        **kwargs will be passed through entrypoint fn to ``timm.models.build_model_with_cfg()``
        and then the model class __init__(). kwargs values set to None are pruned before passing.
    </Tip>

    Args:
        model_name: Name of model to instantiate.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        pretrained_cfg: Pass in an external pretrained_cfg for model.
        pretrained_cfg_overlay: Replace key-values in base pretrained_cfg with these.
        checkpoint_path: Path of checkpoint to load _after_ the model is initialized.
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
    ```
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == 'hf-hub':
        assert not pretrained_cfg, 'pretrained_cfg should not be set when sourcing model from Hugging Face Hub.'
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
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
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
