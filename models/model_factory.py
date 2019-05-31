from models.inception_v4 import *
from models.inception_resnet_v2 import *
from models.densenet import *
from models.resnet import *
from models.dpn import *
from models.senet import *
from models.xception import *
from models.pnasnet import *
from models.gen_efficientnet import *
from models.inception_v3 import *
from models.gluon_resnet import *

from models.helpers import load_checkpoint


def create_model(
        model_name='resnet50',
        pretrained=None,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        **kwargs):

    margs = dict(num_classes=num_classes, in_chans=in_chans, pretrained=pretrained)

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

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    return model
