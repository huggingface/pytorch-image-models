from .registry import is_model, is_model_in_modules, model_entrypoint
from .helpers import load_checkpoint, load_hf_checkpoint_config
from .layers import set_layer_config


def create_model(
        model_name,
        pretrained=False,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    model_args = dict(pretrained=pretrained)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        if is_model(model_name):
            create_fn = model_entrypoint(model_name)
            model = create_fn(**model_args, **kwargs)
        else:
            try:
                model_cfg = load_hf_checkpoint_config(model_name, revision=kwargs.get("hf_revision"))
                create_fn = model_entrypoint(model_cfg.pop("architecture"))
                model = create_fn(**model_args, **kwargs)
                # Probably need some extra stuff, but this is a PoC of how the config in the model hub
                # could overwrite the default config values.
                model.default_cfg.update(model_cfg)
            except Exception as e:
                raise RuntimeError('Unknown model or checkpoint from the Hugging Face hub (%s)' % model_name)
 

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
