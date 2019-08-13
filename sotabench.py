from torchbench.image_classification import ImageNet
from timm import create_model, list_models
from timm.data import resolve_data_config, create_transform

NUM_GPU = 1
BATCH_SIZE = 256 * NUM_GPU


def _attrib(paper_model_name='', paper_arxiv_id='', batch_size=BATCH_SIZE):
    return dict(
        paper_model_name=paper_model_name,
        paper_arxiv_id=paper_arxiv_id,
        batch_size=batch_size)

model_map = dict(
    #adv_inception_v3=_attrib(paper_model_name='Adversarial Inception V3', paper_arxiv_id=),
    #densenet121=_attrib(paper_model_name=, paper_arxiv_id=), # same weights as torchvision
    #densenet161=_attrib(paper_model_name=, paper_arxiv_id=), # same weights as torchvision
    #densenet169=_attrib(paper_model_name=, paper_arxiv_id=), # same weights as torchvision
    #densenet201=_attrib(paper_model_name=, paper_arxiv_id=), # same weights as torchvision
    dpn68=_attrib(
        paper_model_name='DPN-68', paper_arxiv_id='1707.01629'),
    dpn68b=_attrib(
        paper_model_name='DPN-68b', paper_arxiv_id='1707.01629'),
    dpn92=_attrib(
        paper_model_name='DPN-92', paper_arxiv_id='1707.01629'),
    dpn98=_attrib(
        paper_model_name='DPN-98', paper_arxiv_id='1707.01629'),
    dpn107=_attrib(
        paper_model_name='DPN-107', paper_arxiv_id='1707.01629'),
    dpn131=_attrib(
        paper_model_name='DPN-131', paper_arxiv_id='1707.01629'),
    efficientnet_b0=_attrib(
        paper_model_name='EfficientNet-B0', paper_arxiv_id='1905.11946'),
    efficientnet_b1=_attrib(
        paper_model_name='EfficientNet-B1', paper_arxiv_id='1905.11946'),
    efficientnet_b2=_attrib(
        paper_model_name='EfficientNet-B2', paper_arxiv_id='1905.11946'),
    #ens_adv_inception_resnet_v2=_attrib(paper_model_name=, paper_arxiv_id=),
    fbnetc_100=_attrib(
        paper_model_name='FBNet-C', paper_arxiv_id='1812.03443'),
    gluon_inception_v3=_attrib(
        paper_model_name='Inception V3', paper_arxiv_id='1512.00567'),
    gluon_resnet18_v1b=_attrib(
        paper_model_name='ResNet-18', paper_arxiv_id='1812.01187'),
    gluon_resnet34_v1b=_attrib(
        paper_model_name='ResNet-34', paper_arxiv_id='1812.01187'),
    gluon_resnet50_v1b=_attrib(
        paper_model_name='ResNet-50', paper_arxiv_id='1812.01187'),
    gluon_resnet50_v1c=_attrib(
        paper_model_name='ResNet-50-C', paper_arxiv_id='1812.01187'),
    gluon_resnet50_v1d=_attrib(
        paper_model_name='ResNet-50-D', paper_arxiv_id='1812.01187'),
    gluon_resnet50_v1s=_attrib(
        paper_model_name='ResNet-50-S', paper_arxiv_id='1812.01187'),
    gluon_resnet101_v1b=_attrib(
        paper_model_name='ResNet-101', paper_arxiv_id='1812.01187'),
    gluon_resnet101_v1c=_attrib(
        paper_model_name='ResNet-101-C', paper_arxiv_id='1812.01187'),
    gluon_resnet101_v1d=_attrib(
        paper_model_name='ResNet-101-D', paper_arxiv_id='1812.01187'),
    gluon_resnet101_v1s=_attrib(
        paper_model_name='ResNet-101-S', paper_arxiv_id='1812.01187'),
    gluon_resnet152_v1b=_attrib(
        paper_model_name='ResNet-152', paper_arxiv_id='1812.01187'),
    gluon_resnet152_v1c=_attrib(
        paper_model_name='ResNet-152-C', paper_arxiv_id='1812.01187'),
    gluon_resnet152_v1d=_attrib(
        paper_model_name='ResNet-152-D', paper_arxiv_id='1812.01187'),
    gluon_resnet152_v1s=_attrib(
        paper_model_name='ResNet-152-S', paper_arxiv_id='1812.01187'),
    gluon_resnext50_32x4d=_attrib(
        paper_model_name='ResNeXt-50 32x4d', paper_arxiv_id='1812.01187'),
    gluon_resnext101_32x4d=_attrib(
        paper_model_name='ResNeXt-101 32x4d', paper_arxiv_id='1812.01187'),
    gluon_resnext101_64x4d=_attrib(
        paper_model_name='ResNeXt-101 64x4d', paper_arxiv_id='1812.01187'),
    gluon_senet154=_attrib(
        paper_model_name='SENet-154', paper_arxiv_id='1812.01187'),
    gluon_seresnext50_32x4d=_attrib(
        paper_model_name='SE-ResNeXt-50 32x4d', paper_arxiv_id='1812.01187'),
    gluon_seresnext101_32x4d=_attrib(
        paper_model_name='SE-ResNeXt-101 32x4d', paper_arxiv_id='1812.01187'),
    gluon_seresnext101_64x4d=_attrib(
        paper_model_name='SE-ResNeXt-101 64x4d', paper_arxiv_id='1812.01187'),
    gluon_xception65=_attrib(
        paper_model_name='Modified Aligned Xception', paper_arxiv_id='1802.02611', batch_size=BATCH_SIZE//2),
    ig_resnext101_32x8d=_attrib(
        paper_model_name='ResNeXt-101 32×8d', paper_arxiv_id='1805.00932'),
    ig_resnext101_32x16d=_attrib(
        paper_model_name='ResNeXt-101 32×16d', paper_arxiv_id='1805.00932'),
    ig_resnext101_32x32d=_attrib(
        paper_model_name='ResNeXt-101 32×32d', paper_arxiv_id='1805.00932', batch_size=BATCH_SIZE//2),
    ig_resnext101_32x48d=_attrib(
        paper_model_name='ResNeXt-101 32×48d', paper_arxiv_id='1805.00932', batch_size=BATCH_SIZE//4),
    inception_resnet_v2=_attrib(
        paper_model_name='Inception ResNet V2', paper_arxiv_id='1602.07261'),
    #inception_v3=dict(paper_model_name='Inception V3', paper_arxiv_id=),  # same weights as torchvision
    inception_v4=_attrib(
        paper_model_name='Inception V4', paper_arxiv_id='1602.07261'),
    mixnet_l=_attrib(
        paper_model_name='MixNet-L', paper_arxiv_id='1907.09595'),
    mixnet_m=_attrib(
        paper_model_name='MixNet-M', paper_arxiv_id='1907.09595'),
    mixnet_s=_attrib(
        paper_model_name='MixNet-S', paper_arxiv_id='1907.09595'),
    mnasnet_100=_attrib(
        paper_model_name='MnasNet-B1', paper_arxiv_id='1807.11626'),
    mobilenetv3_100=_attrib(
        paper_model_name='MobileNet V3(1.0)', paper_arxiv_id='1905.02244'),
    nasnetalarge=_attrib(
        paper_model_name='NASNet-A Large', paper_arxiv_id='1707.07012', batch_size=BATCH_SIZE//4),
    pnasnet5large=_attrib(
        paper_model_name='PNASNet-5', paper_arxiv_id='1712.00559', batch_size=BATCH_SIZE//4),
    resnet18=_attrib(
        paper_model_name='ResNet-18', paper_arxiv_id='1812.01187'),
    resnet26=_attrib(
        paper_model_name='ResNet-26', paper_arxiv_id='1812.01187'),
    resnet26d=_attrib(
        paper_model_name='ResNet-26-D', paper_arxiv_id='1812.01187'),
    resnet34=_attrib(
        paper_model_name='ResNet-34', paper_arxiv_id='1812.01187'),
    resnet50=_attrib(
        paper_model_name='ResNet-50', paper_arxiv_id='1812.01187'),
    #resnet101=_attrib(paper_model_name=, paper_arxiv_id=),  # same weights as torchvision
    #resnet152=_attrib(paper_model_name=, paper_arxiv_id=),  # same weights as torchvision
    resnext50_32x4d=_attrib(
        paper_model_name='ResNeXt-50 32x4d', paper_arxiv_id='1812.01187'),
    resnext50d_32x4d=_attrib(
        paper_model_name='ResNeXt-50-D 32x4d', paper_arxiv_id='1812.01187'),
    #resnext101_32x8d=_attrib(paper_model_name=, paper_arxiv_id=),  # same weights as torchvision
    semnasnet_100=_attrib(
        paper_model_name='MnasNet-A1', paper_arxiv_id='1807.11626'),
    senet154=_attrib(
        paper_model_name='SENet-154', paper_arxiv_id='1709.01507'),
    seresnet18=_attrib(
        paper_model_name='SE-ResNet-18', paper_arxiv_id='1709.01507'),
    seresnet34=_attrib(
        paper_model_name='SE-ResNet-34', paper_arxiv_id='1709.01507'),
    seresnet50=_attrib(
        paper_model_name='SE-ResNet-50', paper_arxiv_id='1709.01507'),
    seresnet101=_attrib(
        paper_model_name='SE-ResNet-101', paper_arxiv_id='1709.01507'),
    seresnet152=_attrib(
        paper_model_name='SE-ResNet-152', paper_arxiv_id='1709.01507'),
    seresnext26_32x4d=_attrib(
        paper_model_name='SE-ResNeXt-26 32x4d', paper_arxiv_id='1709.01507'),
    seresnext50_32x4d=_attrib(
        paper_model_name='SE-ResNeXt-50 32x4d', paper_arxiv_id='1709.01507'),
    seresnext101_32x4d=_attrib(
        paper_model_name='SE-ResNeXt-101 32x4d', paper_arxiv_id='1709.01507'),
    spnasnet_100=_attrib(
        paper_model_name='Single-Path NAS', paper_arxiv_id='1904.02877'),
    tf_efficientnet_b0=_attrib(
        paper_model_name='EfficientNet-B0', paper_arxiv_id='1905.11946'),
    tf_efficientnet_b1=_attrib(
        paper_model_name='EfficientNet-B1', paper_arxiv_id='1905.11946'),
    tf_efficientnet_b2=_attrib(
        paper_model_name='EfficientNet-B2', paper_arxiv_id='1905.11946'),
    tf_efficientnet_b3=_attrib(
        paper_model_name='EfficientNet-B3', paper_arxiv_id='1905.11946', batch_size=BATCH_SIZE//2),
    tf_efficientnet_b4=_attrib(
        paper_model_name='EfficientNet-B4', paper_arxiv_id='1905.11946', batch_size=BATCH_SIZE//2),
    tf_efficientnet_b5=_attrib(
        paper_model_name='EfficientNet-B5', paper_arxiv_id='1905.11946', batch_size=BATCH_SIZE//4),
    tf_efficientnet_b6=_attrib(
        paper_model_name='EfficientNet-B6', paper_arxiv_id='1905.11946', batch_size=BATCH_SIZE//8),
    tf_efficientnet_b7=_attrib(
        paper_model_name='EfficientNet-B7', paper_arxiv_id='1905.11946', batch_size=BATCH_SIZE//8),
    tf_inception_v3=_attrib(
        paper_model_name='Inception V3', paper_arxiv_id='1512.00567'),
    tf_mixnet_l=_attrib(
        paper_model_name='MixNet-L', paper_arxiv_id='1907.09595'),
    tf_mixnet_m=_attrib(
        paper_model_name='MixNet-M', paper_arxiv_id='1907.09595'),
    tf_mixnet_s=_attrib(
        paper_model_name='MixNet-S', paper_arxiv_id='1907.09595'),
    #tv_resnet34=_attrib(paper_model_name=, paper_arxiv_id=), # same weights as torchvision
    #tv_resnet50=_attrib(paper_model_name=, paper_arxiv_id=), # same weights as torchvision
    #tv_resnext50_32x4d=_attrib(paper_model_name=, paper_arxiv_id=), # same weights as torchvision
    #wide_resnet50_2=_attrib(paper_model_name=, paper_arxiv_id=),   # same weights as torchvision
    #wide_resnet101_2=_attrib(paper_model_name=, paper_arxiv_id=),  # same weights as torchvision
    xception=_attrib(
        paper_model_name='Xception', paper_arxiv_id='1610.02357'),
)

model_names = list_models(pretrained=True)

for model_name in model_names:
    if model_name not in model_map:
        print('Skipping %s' % model_name)
        continue

    # create model from name
    model = create_model(model_name, pretrained=True)
    param_count = sum([m.numel() for m in model.parameters()])
    print('Model %s created, param count: %d' % (model_name, param_count))

    # get appropriate transform for model's default pretrained config
    data_config = resolve_data_config(dict(), model=model, verbose=True)
    input_transform = create_transform(**data_config)

    # Run the benchmark
    ImageNet.benchmark(
        model=model,
        paper_model_name=model_map[model_name]['paper_model_name'],
        paper_arxiv_id=model_map[model_name]['paper_arxiv_id'],
        input_transform=input_transform,
        batch_size=model_map[model_name]['batch_size'],
        num_gpu=NUM_GPU,
        #data_root=DATA_ROOT
    )


