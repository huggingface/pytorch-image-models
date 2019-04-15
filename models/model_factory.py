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
from models.mnasnet import mnasnet0_50, mnasnet0_75, mnasnet1_00, mnasnet1_40,\
    semnasnet0_50, semnasnet0_75, semnasnet1_00, semnasnet1_40, mnasnet_small

from models.helpers import load_checkpoint


def create_model(
        model_name='resnet50',
        pretrained=None,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        **kwargs):

    margs = dict(num_classes=num_classes, in_chans=in_chans, pretrained=pretrained)

    if model_name in globals():
        create_fn = globals()[model_name]
        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    return model
