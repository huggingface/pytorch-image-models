from models.inception_v4 import inception_v4
from models.inception_resnet_v2 import inception_resnet_v2
from models.densenet import densenet161, densenet121, densenet169, densenet201
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
    resnext50_32x4d, resnext101_32x4d, resnext101_64x4d, resnext152_32x4d
from models.dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
from models.senet import seresnet18, seresnet34, seresnet50, seresnet101, seresnet152, \
    seresnext26_32x4d, seresnext50_32x4d, seresnext101_32x4d
from models.xception import xception
from models.pnasnet import pnasnet5large
from models.genmobilenet import \
    mnasnet0_50, mnasnet0_75, mnasnet1_00, mnasnet1_40,\
    semnasnet0_50, semnasnet0_75, semnasnet1_00, semnasnet1_40, mnasnet_small,\
    mobilenetv1_1_00, mobilenetv2_1_00, fbnetc_1_00, chamnetv1_1_00, chamnetv2_1_00

from models.helpers import load_checkpoint


def _is_genmobilenet(name):
    genmobilenets = ['mnasnet', 'semnasnet', 'fbnet', 'chamnet', 'mobilenet']
    if any([name.startswith(x) for x in genmobilenets]):
        return True
    return False


def create_model(
        model_name='resnet50',
        pretrained=None,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        **kwargs):

    margs = dict(num_classes=num_classes, in_chans=in_chans, pretrained=pretrained)

    # Not all models have support for batchnorm params passed as args, only genmobilenet variants
    # FIXME better way to do this without pushing support into every other model fn?
    supports_bn_params = _is_genmobilenet(model_name)
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
