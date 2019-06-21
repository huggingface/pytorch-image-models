from .inception_v4 import *
from .inception_resnet_v2 import *
from .densenet import *
from .resnet import *
from .dpn import *
from .senet import *
from .xception import *
from .pnasnet import *
from .gen_efficientnet import *
from .inception_v3 import *
from .gluon_resnet import *

from .helpers import load_checkpoint


def create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        **kwargs):

    margs = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Not all models have support for batchnorm params passed as args, only gen_efficientnet variants
    supports_bn_params = model_name in gen_efficientnet_model_names()
    if not supports_bn_params and any([x in kwargs for x in ['bn_tf', 'bn_momentum', 'bn_eps']]):
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    if model_name in globals():
        create_fn = globals()[model_name]
        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
