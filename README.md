# PyTorch Image Models, etc

## What's New

### Aug 1, 2020
Universal feature extraction, new models, new weights, new test sets.
* All models support the `features_only=True` argument for `create_model` call to return a network that extracts features from the deepest layer at each stride.
* New models
  * CSPResNet, CSPResNeXt, CSPDarkNet, DarkNet
  * ReXNet
  * (Aligned) Xception41/65/71 (a proper port of TF models)
* New trained weights
  * SEResNet50 - 80.3
  * CSPDarkNet53 - 80.1 top-1
  * CSPResNeXt50 - 80.0 to-1
  * DPN68b - 79.2 top-1
  * EfficientNet-Lite0 (non-TF ver) - 75.5 (trained by @hal-314)
* Add 'real' labels for ImageNet and ImageNet-Renditions test set, see [`results/README.md`](results/README.md)

### June 11, 2020
Bunch of changes:
* DenseNet models updated with memory efficient addition from torchvision (fixed a bug), blur pooling and deep stem additions
* VoVNet V1 and V2 models added, 39 V2 variant (ese_vovnet_39b) trained to 79.3 top-1
* Activation factory added along with new activations:
   * select act at model creation time for more flexibility in using activations compatible with scripting or tracing (ONNX export)
   * hard_mish (experimental) added with memory-efficient grad, along with ME hard_swish
   * context mgr for setting exportable/scriptable/no_jit states
* Norm + Activation combo layers added with initial trial support in DenseNet and VoVNet along with impl of EvoNorm and InplaceAbn wrapper that fit the interface
* Torchscript works for all but two of the model types as long as using Pytorch 1.5+, tests added for this
* Some import cleanup and classifier reset changes, all models will have classifier reset to nn.Identity on reset_classifer(0) call
* Prep for 0.1.28 pip release

### May 12, 2020
* Add ResNeSt models (code adapted from https://github.com/zhanghang1989/ResNeSt, paper https://arxiv.org/abs/2004.08955))

### May 3, 2020
* Pruned EfficientNet B1, B2, and B3 (https://arxiv.org/abs/2002.08258) contributed by [Yonathan Aflalo](https://github.com/yoniaflalo)

### May 1, 2020
* Merged a number of execellent contributions in the ResNet model family over the past month
  * BlurPool2D and resnetblur models initiated by [Chris Ha](https://github.com/VRandme), I trained resnetblur50 to 79.3.
  * TResNet models and SpaceToDepth, AntiAliasDownsampleLayer layers by [mrT23](https://github.com/mrT23)
  * ecaresnet (50d, 101d, light) models and two pruned variants using pruning as per (https://arxiv.org/abs/2002.08258) by [Yonathan Aflalo](https://github.com/yoniaflalo)
* 200 pretrained models in total now with updated results csv in results folder

### April 5, 2020
* Add some newly trained MobileNet-V2 models trained with latest h-params, rand augment. They compare quite favourably to EfficientNet-Lite
  * 3.5M param MobileNet-V2 100 @ 73%
  * 4.5M param MobileNet-V2 110d @ 75%
  * 6.1M param MobileNet-V2 140 @ 76.5%
  * 5.8M param MobileNet-V2 120d @ 77.3%

### March 18, 2020
* Add EfficientNet-Lite models w/ weights ported from [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
* Add RandAugment trained ResNeXt-50 32x4d weights with 79.8 top-1. Trained by [Andrew Lavin](https://github.com/andravin) (see Training section for hparams)

## Introduction

Py**T**orch **Im**age **M**odels is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arxiv papers, etc in the README, documentation, and code comments. Please let me know if I missed anything.

## Models

Most included models have pretrained weights. The weights are either from their original sources, ported by myself from their original framework (e.g. Tensorflow models), or trained from scratch using the included training script.

Included models:
* ResNet/ResNeXt (from [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models) with mods by myself)
    * ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNeXt50 (32x4d), ResNeXt101 (32x4d and 64x4d)
    * 'Bag of Tricks' / Gluon C, D, E, S variations (https://arxiv.org/abs/1812.01187)
    * Instagram trained / ImageNet tuned ResNeXt101-32x8d to 32x48d from from [facebookresearch](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/)
    * Res2Net (https://github.com/gasvn/Res2Net, https://arxiv.org/abs/1904.01169)
    * Selective Kernel (SK) Nets (https://arxiv.org/abs/1903.06586)
    * ResNeSt (code adapted from https://github.com/zhanghang1989/ResNeSt, paper https://arxiv.org/abs/2004.08955)
* DLA
    * Original (https://github.com/ucbdrive/dla, https://arxiv.org/abs/1707.06484)
    * Res2Net (https://github.com/gasvn/Res2Net, https://arxiv.org/abs/1904.01169)
* DenseNet (from [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models))
    * DenseNet-121, DenseNet-169, DenseNet-201, DenseNet-161
* Squeeze-and-Excitation ResNet/ResNeXt (from [Cadene](https://github.com/Cadene/pretrained-models.pytorch) with some pretrained weight additions by myself)
    * SENet-154, SE-ResNet-18, SE-ResNet-34, SE-ResNet-50, SE-ResNet-101, SE-ResNet-152, SE-ResNeXt-26 (32x4d), SE-ResNeXt50 (32x4d), SE-ResNeXt101 (32x4d)
* Inception-V3 (from [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models))
* Inception-ResNet-V2 and Inception-V4 (from [Cadene](https://github.com/Cadene/pretrained-models.pytorch) )
* Xception
    * Original Xception from [Cadene](https://github.com/Cadene/pretrained-models.pytorch)
    * MXNet Gluon 'modified aligned' Xception-65 from [Gluon ModelZoo](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo)
    * DeepLab (Aligned) Xception-41, 65, and 71 from [Tensorflow Models](https://github.com/tensorflow/models/tree/master/research/deeplab)
* PNasNet & NASNet-A (from [Cadene](https://github.com/Cadene/pretrained-models.pytorch))
* DPN (from [myself](https://github.com/rwightman/pytorch-dpn-pretrained))
    * DPN-68, DPN-68b, DPN-92, DPN-98, DPN-131, DPN-107
* EfficientNet (from my standalone [GenEfficientNet](https://github.com/rwightman/gen-efficientnet-pytorch)) - A generic model that implements many of the efficient models that utilize similar DepthwiseSeparable and InvertedResidual blocks
    * EfficientNet NoisyStudent (B0-B7, L2) (https://arxiv.org/abs/1911.04252)
    * EfficientNet AdvProp (B0-B8) (https://arxiv.org/abs/1911.09665)
    * EfficientNet (B0-B7) (https://arxiv.org/abs/1905.11946)
    * EfficientNet-EdgeTPU (S, M, L) (https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html)
    * MixNet (https://arxiv.org/abs/1907.09595)
    * MNASNet B1, A1 (Squeeze-Excite), and Small (https://arxiv.org/abs/1807.11626)
    * MobileNet-V2 (https://arxiv.org/abs/1801.04381)    
    * FBNet-C (https://arxiv.org/abs/1812.03443)
    * Single-Path NAS (https://arxiv.org/abs/1904.02877)
* MobileNet-V3 (https://arxiv.org/abs/1905.02244)
* HRNet
    * code from https://github.com/HRNet/HRNet-Image-Classification
    * paper https://arxiv.org/abs/1908.07919
* SelecSLS
    * code from https://github.com/mehtadushy/SelecSLS-Pytorch
    * paper https://arxiv.org/abs/1907.00837
* TResNet
    * code from https://github.com/mrT23/TResNet
    * paper https://arxiv.org/abs/2003.13630
* RegNet
    * paper `Designing Network Design Spaces` - https://arxiv.org/abs/2003.13678
    * reference code at https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
* VovNet V2 (with V1 support)
    * paper `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    * reference code at https://github.com/youngwanLEE/vovnet-detectron2
* CspNet (Cross-Stage Partial Networks)
    * paper `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929
    * reference impl at https://github.com/WongKinYiu/CrossStagePartialNetworks
* ReXNet
    * paper `ReXNet: Diminishing Representational Bottleneck on CNN` - https://arxiv.org/abs/2007.00992
    * code from https://github.com/clovaai/rexnet

Use the  `--model` arg to specify model for train, validation, inference scripts. Match the all lowercase
creation fn for the model you'd like.

## Features
Several (less common) features that I often utilize in my projects are included. Many of their additions are the reason why I maintain my own set of models, instead of using others' via PIP:
* All models have a common default configuration interface and API for
    * accessing/changing the classifier - `get_classifier` and `reset_classifier`
    * doing a forward pass on just the features - `forward_features`
    * these makes it easy to write consistent network wrappers that work with any of the models
* All models support multi-scale feature map extraction (feature pyramids) via create_model
    * `create_model(name, features_only=True, out_indices=..., output_stride=...)`
    * `out_indices` creation arg specifies which feature maps to return, these indices are 0 based and generally correspond to the `C(i + 1)` feature level. Most models start with stride 2 features (`C1`) at index 0 and end with `C5` at index 4. Some models start with stride 1 or 4 and end with 6 (stride 64).
    * `output_stride` creation arg controls output stride of the network, most networks are stride 32 by default. Dilated convs are used to limit the output stride. Not all networks support this.
    * feature map channel counts, reduction level (stride) can be queried AFTER model creation via the `.feature_info` member
* All models have a consistent pretrained weight loader that adapts last linear if necessary, and from 3 to 1 channel input if desired
* High performance [reference training, validation, and inference scripts](https://rwightman.github.io/pytorch-image-models/scripts/) that work in several process/GPU modes:
    * NVIDIA DDP w/ a single GPU per process, multiple processes with APEX present (AMP mixed-precision optional)
    * PyTorch DistributedDataParallel w/ multi-gpu, single process (AMP disabled as it crashes when enabled)
    * PyTorch w/ single GPU single process (AMP optional)
* A dynamic global pool implementation that allows selecting from average pooling, max pooling, average + max, or concat([average, max]) at model creation. All global pooling is adaptive average by default and compatible with pretrained weights.
* A 'Test Time Pool' wrapper that can wrap any of the included models and usually provide improved performance doing inference with input images larger than the training size. Idea adapted from original DPN implementation when I ported (https://github.com/cypw/DPNs)
* Learning rate schedulers
  * Ideas adopted from
     * [AllenNLP schedulers](https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers)
     * [FAIRseq lr_scheduler](https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler)
     * SGDR: Stochastic Gradient Descent with Warm Restarts (https://arxiv.org/abs/1608.03983)
  * Schedulers include `step`, `cosine` w/ restarts, `tanh` w/ restarts, `plateau`
* Optimizers:
    * `rmsprop_tf` adapted from PyTorch RMSProp by myself. Reproduces much improved Tensorflow RMSProp behaviour.
    * `radam` by [Liyuan Liu](https://github.com/LiyuanLucasLiu/RAdam) (https://arxiv.org/abs/1908.03265)
    * `novograd` by [Masashi Kimura](https://github.com/convergence-lab/novograd) (https://arxiv.org/abs/1905.11286)
    * `lookahead` adapted from impl by [Liam](https://github.com/alphadl/lookahead.pytorch) (https://arxiv.org/abs/1907.08610)
    * `fused<name>` optimizers by name with [NVIDIA Apex](https://github.com/NVIDIA/apex/tree/master/apex/optimizers) installed
    * `adamp` and `sgdp` by [Naver ClovAI](https://github.com/clovaai) (https://arxiv.org/abs/2006.08217)
* Random Erasing from [Zhun Zhong](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py)  (https://arxiv.org/abs/1708.04896)
* Mixup (as in https://arxiv.org/abs/1710.09412) - currently implementing/testing
* An inference script that dumps output to CSV is provided as an example
* AutoAugment (https://arxiv.org/abs/1805.09501) and RandAugment (https://arxiv.org/abs/1909.13719) ImageNet configurations modeled after impl for EfficientNet training (https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py)
* AugMix w/ JSD loss (https://arxiv.org/abs/1912.02781), JSD w/ clean + augmented mixing support works with AutoAugment and RandAugment as well
* SplitBachNorm - allows splitting batch norm layers between clean and augmented (auxiliary batch norm) data
* DropPath aka "Stochastic Depth" (https://arxiv.org/abs/1603.09382) 
* DropBlock (https://arxiv.org/abs/1810.12890)
* Efficient Channel Attention - ECA (https://arxiv.org/abs/1910.03151)
* Blur Pooling (https://arxiv.org/abs/1904.11486)

## Results

Model validation results can be found in the [documentation](https://rwightman.github.io/pytorch-image-models/results/) and in the [results tables](results/README.md)

## Getting Started

See [documentation](https://rwightman.github.io/pytorch-image-models/)
