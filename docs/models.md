# Model Architectures

__FIXME - Clean This Up!__

### ResNet / ResNeXt

* ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNeXt50 (32x4d), ResNeXt101 (32x4d and 64x4d) 
* 'Bag of Tricks' / Gluon C, D, E, S variations (https://arxiv.org/abs/1812.01187)
* Instagram trained / ImageNet tuned ResNeXt101-32x8d to 32x48d from from [facebookresearch](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/)
* Res2Net (https://github.com/gasvn/Res2Net, https://arxiv.org/abs/1904.01169)
* Selective Kernel (SK) Nets (https://arxiv.org/abs/1903.06586)
* ResNeSt (code adapted from https://github.com/zhanghang1989/ResNeSt, paper https://arxiv.org/abs/2004.08955)

Originally based on ResNet from [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models)

### DLA

* Original
  * code: https://github.com/ucbdrive/dla
  * paper: https://arxiv.org/abs/1707.06484
* Res2Net
  * code: https://github.com/gasvn/Res2Net
  * paper: https://arxiv.org/abs/1904.01169

### DenseNet 

* DenseNet-121, DenseNet-169, DenseNet-201, DenseNet-161

Code from [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models)

### Squeeze-and-Excitation ResNet/ResNeXt 

* SENet-154, SE-ResNet-18, SE-ResNet-34, SE-ResNet-50, SE-ResNet-101, SE-ResNet-152, SE-ResNeXt-26 (32x4d), SE-ResNeXt50 (32x4d), SE-ResNeXt101 (32x4d)

Code from [Cadene pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) with modifications

### Inception-V3 

Code from [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models)

### Inception-ResNet-V2 and Inception-V4 

Code from [Cadene pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

### Xception and Aligned-Xception (DeepLab)

* Original variant from [Cadene pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
* MXNet Gluon 'modified aligned' Xception-65 and 71 models from [Gluon ModelZoo](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo)
* DeepLab (Aligned) Xception-41, 65, and 71 from [Tensorflow Models](https://github.com/tensorflow/models/tree/master/research/deeplab)

### PNasNet & NASNet-A

Code from [Cadene pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
 
### DPN

* DPN-68, DPN-68b, DPN-92, DPN-98, DPN-131, DPN-107 

Code adapted by [myself](https://github.com/rwightman/pytorch-dpn-pretrained) from MXNet originals (https://github.com/cypw/DPNs)

### EfficientNet 

* EfficientNet NoisyStudent (B0-B7, L2) (https://arxiv.org/abs/1911.04252)
* EfficientNet AdvProp (B0-B8) (https://arxiv.org/abs/1911.09665)
* EfficientNet (B0-B7) (https://arxiv.org/abs/1905.11946)
* EfficientNet-EdgeTPU (S, M, L) (https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html)
* MixNet (https://arxiv.org/abs/1907.09595)
* MNASNet B1, A1 (Squeeze-Excite), and Small (https://arxiv.org/abs/1807.11626)
* MobileNet-V2 (https://arxiv.org/abs/1801.04381)    
* FBNet-C (https://arxiv.org/abs/1812.03443)
* Single-Path NAS (https://arxiv.org/abs/1904.02877)

Code from my standalone [GenEfficientNet](https://github.com/rwightman/gen-efficientnet-pytorch), adapted from [Tensorflow originals](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

### MobileNet-V3
 
* MobileNetV3-Large, MobileNetV3-Small (https://arxiv.org/abs/1905.02244)

Code from my standalone [GenEfficientNet](https://github.com/rwightman/gen-efficientnet-pytorch), adapted from [Tensorflow originals](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

### HRNet

* code from https://github.com/HRNet/HRNet-Image-Classification
* paper https://arxiv.org/abs/1908.07919

### SelecSLS

* paper https://arxiv.org/abs/1907.00837
* code from https://github.com/mehtadushy/SelecSLS-Pytorch

### TResNet

* paper https://arxiv.org/abs/2003.13630
* code from https://github.com/mrT23/TResNet

### RegNet

* paper `Designing Network Design Spaces` - https://arxiv.org/abs/2003.13678
* reference code at https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py

### VovNet V2 / V1

* paper `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
* reference code at https://github.com/youngwanLEE/vovnet-detectron2

### CspNet (Cross-Stage Partial Networks)
* paper `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929
* reference impl at https://github.com/WongKinYiu/CrossStagePartialNetworks

### ReXNet
* paper `ReXNet: Diminishing Representational Bottleneck on CNN` - https://arxiv.org/abs/2007.00992
* code from https://github.com/clovaai/rexnet
