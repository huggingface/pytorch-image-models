import torch
from torchvision import transforms
from PIL import Image
import math
import os

from .inception_v4 import inception_v4
from .inception_resnet_v2 import inception_resnet_v2
from .wrn50_2 import wrn50_2
from .my_densenet import densenet161, densenet121, densenet169, densenet201
from .my_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .fbresnet200 import fbresnet200
from .dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
from .senet import se_resnet18, se_resnet34, se_resnet50, se_resnet101, se_resnet152,\
    se_resnext50_32x4d, se_resnext101_32x4d


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
        'model_name': 'inception_resnet_v2', 'num_classes': 1001, 'input_size': 299, 'normalizer': 'le'},
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
        model = inception_v4(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'wrn50':
        model = wrn50_2(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'fbresnet200':
        model = fbresnet200(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'seresnet18':
        model = se_resnet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'seresnet34':
        model = se_resnet34(num_classes=num_classes, pretrained=pretrained)
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


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


DEFAULT_CROP_PCT = 0.875


def get_transforms_train(model_name, img_size=224):
    if 'dpn' in model_name:
        normalize = transforms.Normalize(
            mean=[124 / 255, 117 / 255, 104 / 255],
            std=[1 / (.0167 * 255)] * 3)
    elif 'inception' in model_name:
        normalize = LeNormalize()
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        normalize])


def get_transforms_eval(model_name, img_size=224, crop_pct=None):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    if 'dpn' in model_name:
        if crop_pct is None:
            # Use default 87.5% crop for model's native img_size
            # but use 100% crop for larger than native as it
            # improves test time results across all models.
            if img_size == 224:
                scale_size = int(math.floor(img_size / DEFAULT_CROP_PCT))
            else:
                scale_size = img_size
        else:
            scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=[124 / 255, 117 / 255, 104 / 255],
            std=[1 / (.0167 * 255)] * 3)
    elif 'inception' in model_name:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = LeNormalize()
    else:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize])
