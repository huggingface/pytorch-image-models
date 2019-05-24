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
    mnasnet_050, mnasnet_075, mnasnet_100, mnasnet_140, tflite_mnasnet_100,\
    semnasnet_050, semnasnet_075, semnasnet_100, semnasnet_140, tflite_semnasnet_100, mnasnet_small,\
    mobilenetv1_100, mobilenetv2_100, mobilenetv3_050, mobilenetv3_075, mobilenetv3_100,\
    fbnetc_100, chamnetv1_100, chamnetv2_100, spnasnet_100
from models.inception_v3 import inception_v3, gluon_inception_v3, tf_inception_v3, adv_inception_v3
from models.gluon_resnet import gluon_resnet18_v1b, gluon_resnet34_v1b, gluon_resnet50_v1b, gluon_resnet101_v1b, \
    gluon_resnet152_v1b, gluon_resnet50_v1c, gluon_resnet101_v1c, gluon_resnet152_v1c, \
    gluon_resnet50_v1d, gluon_resnet101_v1d, gluon_resnet152_v1d, \
    gluon_resnet50_v1e, gluon_resnet101_v1e, gluon_resnet152_v1e, \
    gluon_resnet50_v1s, gluon_resnet101_v1s, gluon_resnet152_v1s, \
    gluon_resnext50_32x4d, gluon_resnext101_32x4d , gluon_resnext101_64x4d, gluon_resnext152_32x4d, \
    gluon_seresnext50_32x4d, gluon_seresnext101_32x4d, gluon_seresnext101_64x4d, gluon_seresnext152_32x4d, \
    gluon_senet154

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
