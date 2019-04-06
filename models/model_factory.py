import torch
import os

from .inception_v4 import inception_v4
from .inception_resnet_v2 import inception_resnet_v2
from .densenet import densenet161, densenet121, densenet169, densenet201
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .fbresnet200 import fbresnet200
from .dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
from .senet import seresnet18, seresnet34, seresnet50, seresnet101, seresnet152, \
    seresnext26_32x4d, seresnext50_32x4d, seresnext101_32x4d
from .resnext import resnext50, resnext101, resnext152
from .xception import xception
from .pnasnet import pnasnet5large

model_config_dict = {
    'resnet18': {
        'model_name': 'resnet18', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'resnet34': {
        'model_name': 'resnet34', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'resnet50': {
        'model_name': 'resnet50', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'resnet101': {
        'model_name': 'resnet101', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'resnet152': {
        'model_name': 'resnet152', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'densenet121': {
        'model_name': 'densenet121', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'densenet169': {
        'model_name': 'densenet169', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'densenet201': {
        'model_name': 'densenet201', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'densenet161': {
        'model_name': 'densenet161', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'tv'},
    'dpn107': {
        'model_name': 'dpn107', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dpn'},
    'dpn92_extra': {
        'model_name': 'dpn92', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dpn'},
    'dpn92': {
        'model_name': 'dpn92', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dpn'},
    'dpn68': {
        'model_name': 'dpn68', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dpn'},
    'dpn68b': {
        'model_name': 'dpn68b', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dpn'},
    'dpn68b_extra': {
        'model_name': 'dpn68b', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dpn'},
    'inception_resnet_v2': {
        'model_name': 'inception_resnet_v2', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'le'},
    'xception': {
        'model_name': 'xception', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'le'},
    'pnasnet5large': {
        'model_name': 'pnasnet5large', 'num_classes': 1000, 'input_size': 331, 'normalizer': 'le'}
}


def create_model(
        model_name='resnet50',
        pretrained=None,
        num_classes=1000,
        checkpoint_path='',
        **kwargs):

    test_time_pool = kwargs.pop('test_time_pool') if 'test_time_pool' in kwargs else 0

    if model_name == 'dpn68':
        model = dpn68(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn68b':
        model = dpn68b(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn92':
        model = dpn92(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn98':
        model = dpn98(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn131':
        model = dpn131(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn107':
        model = dpn107(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet161':
        model = densenet161(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet169':
        model = densenet169(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet201':
        model = densenet201(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'inception_resnet_v2':
        model = inception_resnet_v2(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'inception_v4':
        model = inception_v4(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'fbresnet200':
        model = fbresnet200(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'seresnet18':
        model = seresnet18(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'seresnet34':
        model = seresnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'seresnet50':
        model = seresnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'seresnet101':
        model = seresnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'seresnet152':
        model = seresnet152(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'seresnext26_32x4d':
        model = seresnext26_32x4d(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'seresnext50_32x4d':
        model = seresnext50_32x4d(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'seresnext101_32x4d':
        model = seresnext101_32x4d(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnext50':
        model = resnext50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnext101':
        model = resnext101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnext152':
        model = resnext152(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'xception':
        model = xception(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'pnasnet5large':
        model = pnasnet5large(num_classes=num_classes, pretrained=pretrained)
    else:
        assert False and "Invalid model"

    if checkpoint_path and not pretrained:
        print(checkpoint_path)
        load_checkpoint(model, checkpoint_path)

    return model


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print('Loading checkpoint', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: No checkpoint found at %s." % checkpoint_path)

