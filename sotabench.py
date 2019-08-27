from torchbench.image_classification import ImageNet
from timm import create_model
from timm.data import resolve_data_config, create_transform
from timm.models import TestTimePoolHead
import os

NUM_GPU = 1
BATCH_SIZE = 256 * NUM_GPU


def _entry(model_name, paper_model_name, paper_arxiv_id, batch_size=BATCH_SIZE, ttp=False, args=dict()):
    return dict(
        model=model_name,
        paper_model_name=paper_model_name,
        paper_arxiv_id=paper_arxiv_id,
        batch_size=batch_size,
        ttp=ttp,
        args=args)


model_list = [
    #_entry('adv_inception_v3', 'Adversarial Inception V3', ),
    #_entry('densenet121'), # same weights as torchvision
    #_entry('densenet161'), # same weights as torchvision
    #_entry('densenet169'), # same weights as torchvision
    #_entry('densenet201'), # same weights as torchvision
    _entry('dpn68', 'DPN-68 (224x224)', '1707.01629'),
    _entry('dpn68b', 'DPN-68b (224x224)', '1707.01629'),
    _entry('dpn92', 'DPN-92 (224x224)', '1707.01629'),
    _entry('dpn98', 'DPN-98 (224x224)', '1707.01629'),
    _entry('dpn107', 'DPN-107 (224x224)', '1707.01629'),
    _entry('dpn131', 'DPN-131 (224x224)', '1707.01629'),
    _entry('dpn68', 'DPN-68 (320x320, Mean-Max Pooling)', '1707.01629', ttp=True, args=dict(img_size=320)),
    _entry('dpn68b', 'DPN-68b (320x320, Mean-Max Pooling)', '1707.01629', ttp=True, args=dict(img_size=320)),
    _entry('dpn92', 'DPN-92 (320x320, Mean-Max Pooling)', '1707.01629',
           ttp=True, args=dict(img_size=320), batch_size=BATCH_SIZE//2),
    _entry('dpn98', 'DPN-98 (320x320, Mean-Max Pooling)', '1707.01629',
           ttp=True, args=dict(img_size=320), batch_size=BATCH_SIZE//2),
    _entry('dpn107', 'DPN-107 (320x320, Mean-Max Pooling)', '1707.01629',
           ttp=True, args=dict(img_size=320), batch_size=BATCH_SIZE//4),
    _entry('dpn131', 'DPN-131 (320x320, Mean-Max Pooling)', '1707.01629',
           ttp=True, args=dict(img_size=320), batch_size=BATCH_SIZE//4),
    _entry('efficientnet_b0', 'EfficientNet-B0', '1905.11946'),
    _entry('efficientnet_b1', 'EfficientNet-B1', '1905.11946'),
    _entry('efficientnet_b2', 'EfficientNet-B2', '1905.11946'),
    #_entry('ens_adv_inception_resnet_v2', 'Ensemble Adversarial Inception V3'),
    _entry('fbnetc_100', 'FBNet-C', '1812.03443'),
    _entry('gluon_inception_v3', 'Inception V3', '1512.00567'),
    _entry('gluon_resnet18_v1b', 'ResNet-18', '1812.01187'),
    _entry('gluon_resnet34_v1b', 'ResNet-34', '1812.01187'),
    _entry('gluon_resnet50_v1b', 'ResNet-50', '1812.01187'),
    _entry('gluon_resnet50_v1c', 'ResNet-50-C', '1812.01187'),
    _entry('gluon_resnet50_v1d', 'ResNet-50-D', '1812.01187'),
    _entry('gluon_resnet50_v1s', 'ResNet-50-S', '1812.01187'),
    _entry('gluon_resnet101_v1b', 'ResNet-101', '1812.01187'),
    _entry('gluon_resnet101_v1c', 'ResNet-101-C', '1812.01187'),
    _entry('gluon_resnet101_v1d', 'ResNet-101-D', '1812.01187'),
    _entry('gluon_resnet101_v1s', 'ResNet-101-S', '1812.01187'),
    _entry('gluon_resnet152_v1b', 'ResNet-152', '1812.01187'),
    _entry('gluon_resnet152_v1c', 'ResNet-152-C', '1812.01187'),
    _entry('gluon_resnet152_v1d', 'ResNet-152-D', '1812.01187'),
    _entry('gluon_resnet152_v1s', 'ResNet-152-S', '1812.01187'),
    _entry('gluon_resnext50_32x4d', 'ResNeXt-50 32x4d', '1812.01187'),
    _entry('gluon_resnext101_32x4d', 'ResNeXt-101 32x4d', '1812.01187'),
    _entry('gluon_resnext101_64x4d', 'ResNeXt-101 64x4d', '1812.01187'),
    _entry('gluon_senet154', 'SENet-154', '1812.01187'),
    _entry('gluon_seresnext50_32x4d', 'SE-ResNeXt-50 32x4d', '1812.01187'),
    _entry('gluon_seresnext101_32x4d', 'SE-ResNeXt-101 32x4d', '1812.01187'),
    _entry('gluon_seresnext101_64x4d', 'SE-ResNeXt-101 64x4d', '1812.01187'),
    _entry('gluon_xception65', 'Modified Aligned Xception', '1802.02611', batch_size=BATCH_SIZE//2),
    _entry('ig_resnext101_32x8d', 'ResNeXt-101 32x8d', '1805.00932'),
    _entry('ig_resnext101_32x16d', 'ResNeXt-101 32x16d', '1805.00932'),
    _entry('ig_resnext101_32x32d', 'ResNeXt-101 32x32d', '1805.00932', batch_size=BATCH_SIZE//2),
    _entry('ig_resnext101_32x48d', 'ResNeXt-101 32x48d', '1805.00932', batch_size=BATCH_SIZE//4),
    _entry('inception_resnet_v2', 'Inception ResNet V2', '1602.07261'),
    #_entry('inception_v3', paper_model_name='Inception V3', ),  # same weights as torchvision
    _entry('inception_v4', 'Inception V4', '1602.07261'),
    _entry('mixnet_l', 'MixNet-L', '1907.09595'),
    _entry('mixnet_m', 'MixNet-M', '1907.09595'),
    _entry('mixnet_s', 'MixNet-S', '1907.09595'),
    _entry('mnasnet_100', 'MnasNet-B1', '1807.11626'),
    _entry('mobilenetv3_100', 'MobileNet V3(1.0)', '1905.02244'),
    _entry('nasnetalarge', 'NASNet-A Large', '1707.07012', batch_size=BATCH_SIZE//4),
    _entry('pnasnet5large', 'PNASNet-5', '1712.00559', batch_size=BATCH_SIZE//4),
    _entry('resnet18', 'ResNet-18', '1812.01187'),
    _entry('resnet26', 'ResNet-26', '1812.01187'),
    _entry('resnet26d', 'ResNet-26-D', '1812.01187'),
    _entry('resnet34', 'ResNet-34', '1812.01187'),
    _entry('resnet50', 'ResNet-50', '1812.01187'),
    #_entry('resnet101', , ),  # same weights as torchvision
    #_entry('resnet152', , ),  # same weights as torchvision
    _entry('resnext50_32x4d', 'ResNeXt-50 32x4d', '1812.01187'),
    _entry('resnext50d_32x4d', 'ResNeXt-50-D 32x4d', '1812.01187'),
    #_entry('resnext101_32x8d', ),  # same weights as torchvision
    _entry('semnasnet_100', 'MnasNet-A1', '1807.11626'),
    _entry('senet154', 'SENet-154', '1709.01507'),
    _entry('seresnet18', 'SE-ResNet-18', '1709.01507'),
    _entry('seresnet34', 'SE-ResNet-34', '1709.01507'),
    _entry('seresnet50', 'SE-ResNet-50', '1709.01507'),
    _entry('seresnet101', 'SE-ResNet-101', '1709.01507'),
    _entry('seresnet152', 'SE-ResNet-152', '1709.01507'),
    _entry('seresnext26_32x4d', 'SE-ResNeXt-26 32x4d', '1709.01507'),
    _entry('seresnext50_32x4d', 'SE-ResNeXt-50 32x4d', '1709.01507'),
    _entry('seresnext101_32x4d', 'SE-ResNeXt-101 32x4d', '1709.01507'),
    _entry('spnasnet_100', 'Single-Path NAS', '1904.02877'),
    _entry('tf_efficientnet_b0', 'EfficientNet-B0 (AutoAugment)', '1905.11946'),
    _entry('tf_efficientnet_b1', 'EfficientNet-B1 (AutoAugment)', '1905.11946'),
    _entry('tf_efficientnet_b2', 'EfficientNet-B2 (AutoAugment)', '1905.11946'),
    _entry('tf_efficientnet_b3', 'EfficientNet-B3 (AutoAugment)', '1905.11946', batch_size=BATCH_SIZE//2),
    _entry('tf_efficientnet_b4', 'EfficientNet-B4 (AutoAugment)', '1905.11946', batch_size=BATCH_SIZE//2),
    _entry('tf_efficientnet_b5', 'EfficientNet-B5 (AutoAugment)', '1905.11946', batch_size=BATCH_SIZE//4),
    _entry('tf_efficientnet_b6', 'EfficientNet-B6 (AutoAugment)', '1905.11946', batch_size=BATCH_SIZE//8),
    _entry('tf_efficientnet_b7', 'EfficientNet-B7 (AutoAugment)', '1905.11946', batch_size=BATCH_SIZE//8),
    _entry('tf_efficientnet_es', 'EfficientNet-EdgeTPU-S', '1905.11946'),
    _entry('tf_efficientnet_em', 'EfficientNet-EdgeTPU-M', '1905.11946'),
    _entry('tf_efficientnet_el', 'EfficientNet-EdgeTPU-L', '1905.11946', batch_size=BATCH_SIZE//2),
    _entry('tf_inception_v3', 'Inception V3', '1512.00567'),
    _entry('tf_mixnet_l', 'MixNet-L', '1907.09595'),
    _entry('tf_mixnet_m', 'MixNet-M', '1907.09595'),
    _entry('tf_mixnet_s', 'MixNet-S', '1907.09595'),
    #_entry('tv_resnet34', , ), # same weights as torchvision
    #_entry('tv_resnet50', , ), # same weights as torchvision
    #_entry('tv_resnext50_32x4d', , ), # same weights as torchvision
    #_entry('wide_resnet50_2' , ),   # same weights as torchvision
    #_entry('wide_resnet101_2', , ),  # same weights as torchvision
    _entry('xception', 'Xception', '1610.02357'),
]


for m in model_list:
    model_name = m['model']
    # create model from name
    model = create_model(model_name, pretrained=True)
    param_count = sum([m.numel() for m in model.parameters()])
    print('Model %s, %s created. Param count: %d' % (model_name, m['paper_model_name'], param_count))

    # get appropriate transform for model's default pretrained config
    data_config = resolve_data_config(m['args'], model=model, verbose=True)
    if m['ttp']:
        model = TestTimePoolHead(model, model.default_cfg['pool_size'])
        data_config['crop_pct'] = 1.0
    input_transform = create_transform(**data_config)

    # Run the benchmark
    ImageNet.benchmark(
        model=model,
        paper_model_name=m['paper_model_name'],
        paper_arxiv_id=m['paper_arxiv_id'],
        input_transform=input_transform,
        batch_size=m['batch_size'],
        num_gpu=NUM_GPU,
        data_root=os.environ.get('IMAGENET_DIR', './imagenet')
    )


