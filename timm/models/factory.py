from .registry import is_model, is_model_in_modules, model_entrypoint
from .helpers import load_checkpoint


def create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    margs = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Not all models have support for batchnorm params passed as args, only gen_efficientnet variants
    supports_bn_params = is_model_in_modules(model_name, ['gen_efficientnet'])
    if not supports_bn_params and any([x in kwargs for x in ['bn_tf', 'bn_momentum', 'bn_eps']]):
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
