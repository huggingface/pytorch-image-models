from torchvision.models import Inception3
from .registry import register_model
from .helpers import load_pretrained
from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

__all__ = []

default_cfgs = {
    # original PyTorch weights, ported from Tensorflow but modified
    'inception_v3': {
        'url': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
        'input_size': (3, 299, 299),
        'crop_pct': 0.875,
        'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN,  # also works well enough with resnet defaults
        'std': IMAGENET_INCEPTION_STD,  # also works well enough with resnet defaults
        'num_classes': 1000,
        'first_conv': 'conv0',
        'classifier': 'fc'
    },
    # my port of Tensorflow SLIM weights (http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
    'tf_inception_v3': {
        'url': 'https://www.dropbox.com/s/xdh32bpdgqzpx8t/tf_inception_v3-e0069de4.pth?dl=1',
        'input_size': (3, 299, 299),
        'crop_pct': 0.875,
        'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'num_classes': 1001,
        'first_conv': 'conv0',
        'classifier': 'fc'
    },
    # my port of Tensorflow adversarially trained Inception V3 from
    # http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    'adv_inception_v3': {
        'url': 'https://www.dropbox.com/s/b5pudqh84gtl7i8/adv_inception_v3-9e27bd63.pth?dl=1',
        'input_size': (3, 299, 299),
        'crop_pct': 0.875,
        'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'num_classes': 1001,
        'first_conv': 'conv0',
        'classifier': 'fc'
    },
    # from gluon pretrained models, best performing in terms of accuracy/loss metrics
    # https://gluon-cv.mxnet.io/model_zoo/classification.html
    'gluon_inception_v3': {
        'url': 'https://www.dropbox.com/s/8uv6wrl6it6394u/gluon_inception_v3-9f746940.pth?dl=1',
        'input_size': (3, 299, 299),
        'crop_pct': 0.875,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,  # also works well with inception defaults
        'std': IMAGENET_DEFAULT_STD,  # also works well with inception defaults
        'num_classes': 1000,
        'first_conv': 'conv0',
        'classifier': 'fc'
    }
}


def _assert_default_kwargs(kwargs):
    # for imported models (ie torchvision) without capability to change these params,
    # make sure they aren't being set to non-defaults
    assert kwargs.pop('global_pool', 'avg') == 'avg'
    assert kwargs.pop('drop_rate', 0.) == 0.


@register_model
def inception_v3(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    # original PyTorch weights, ported from Tensorflow but modified
    default_cfg = default_cfgs['inception_v3']
    assert in_chans == 3
    _assert_default_kwargs(kwargs)
    model = Inception3(num_classes=num_classes, aux_logits=True, transform_input=False)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    model.default_cfg = default_cfg
    return model


@register_model
def tf_inception_v3(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    # my port of Tensorflow SLIM weights (http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
    default_cfg = default_cfgs['tf_inception_v3']
    assert in_chans == 3
    _assert_default_kwargs(kwargs)
    model = Inception3(num_classes=num_classes, aux_logits=False, transform_input=False)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    model.default_cfg = default_cfg
    return model


@register_model
def adv_inception_v3(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    # my port of Tensorflow adversarially trained Inception V3 from
    # http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    default_cfg = default_cfgs['adv_inception_v3']
    assert in_chans == 3
    _assert_default_kwargs(kwargs)
    model = Inception3(num_classes=num_classes, aux_logits=False, transform_input=False)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    model.default_cfg = default_cfg
    return model


@register_model
def gluon_inception_v3(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    # from gluon pretrained models, best performing in terms of accuracy/loss metrics
    # https://gluon-cv.mxnet.io/model_zoo/classification.html
    default_cfg = default_cfgs['gluon_inception_v3']
    assert in_chans == 3
    _assert_default_kwargs(kwargs)
    model = Inception3(num_classes=num_classes, aux_logits=False, transform_input=False)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    model.default_cfg = default_cfg
    return model
