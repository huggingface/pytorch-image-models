# PyTorch Image Models, etc

## What's New

### Oct 30, 2020
* Test with PyTorch 1.7 and fix a small top-n metric view vs reshape issue.
* Convert newly added 224x224 Vision Transformer weights from official JAX repo. 81.8 top-1 for B/16, 83.1 L/16.
* Support PyTorch 1.7 optimized, native SiLU (aka Swish) activation. Add mapping to 'silu' name, custom swish will eventually be deprecated.
* Fix regression for loading pretrained classifier via direct model entrypoint functions. Didn't impact create_model() factory usage.
* PyPi release @ 0.3.0 version!

### Oct 26, 2020
* Update Vision Transformer models to be compatible with official code release at https://github.com/google-research/vision_transformer
* Add Vision Transformer weights (ImageNet-21k pretrain) for 384x384 base and large models converted from official jax impl
  * ViT-B/16 - 84.2
  * ViT-B/32 - 81.7
  * ViT-L/16 - 85.2
  * ViT-L/32 - 81.5

### Oct 21, 2020
* Weights added for Vision Transformer (ViT) models. 77.86 top-1 for 'small' and 79.35 for 'base'. Thanks to [Christof](https://www.kaggle.com/christofhenkel) for training the base model w/ lots of GPUs.

### Oct 13, 2020
* Initial impl of Vision Transformer models. Both patch and hybrid (CNN backbone) variants. Currently trying to train...
* Adafactor and AdaHessian (FP32 only, no AMP) optimizers
* EdgeTPU-M (`efficientnet_em`) model trained in PyTorch, 79.3 top-1
* Pip release, doc updates pending a few more changes...

### Sept 18, 2020
* New ResNet 'D' weights. 72.7 (top-1) ResNet-18-D, 77.1 ResNet-34-D, 80.5 ResNet-50-D
* Added a few untrained defs for other ResNet models (66D, 101D, 152D, 200/200D)

### Sept 3, 2020
* New weights
  * Wide-ResNet50 - 81.5 top-1 (vs 78.5 torchvision)
  * SEResNeXt50-32x4d - 81.3 top-1 (vs 79.1 cadene)
* Support for native Torch AMP and channels_last memory format added to train/validate scripts (`--channels-last`, `--native-amp` vs `--apex-amp`)
* Models tested with channels_last on latest NGC 20.08 container. AdaptiveAvgPool in attn layers changed to mean((2,3)) to work around bug with NHWC kernel.

### Aug 12, 2020
* New/updated weights from training experiments
  * EfficientNet-B3 - 82.1 top-1 (vs 81.6 for official with AA and 81.9 for AdvProp)
  * RegNetY-3.2GF - 82.0 top-1 (78.9 from official ver)
  * CSPResNet50 - 79.6 top-1 (76.6 from official ver)
* Add CutMix integrated w/ Mixup. See [pull request](https://github.com/rwightman/pytorch-image-models/pull/218) for some usage examples
* Some fixes for using pretrained weights with `in_chans` != 3 on several models.

### Aug 5, 2020
Universal feature extraction, new models, new weights, new test sets.
* All models support the `features_only=True` argument for `create_model` call to return a network that extracts feature maps from the deepest layer at each stride.
* New models
  * CSPResNet, CSPResNeXt, CSPDarkNet, DarkNet
  * ReXNet
  * (Modified Aligned) Xception41/65/71 (a proper port of TF models)
* New trained weights
  * SEResNet50 - 80.3 top-1
  * CSPDarkNet53 - 80.1 top-1
  * CSPResNeXt50 - 80.0 top-1
  * DPN68b - 79.2 top-1
  * EfficientNet-Lite0 (non-TF ver) - 75.5 (submitted by [@hal-314](https://github.com/hal-314))
* Add 'real' labels for ImageNet and ImageNet-Renditions test set, see [`results/README.md`](results/README.md)
* Test set ranking/top-n diff script by [@KushajveerSingh](https://github.com/KushajveerSingh)
* Train script and loader/transform tweaks to punch through more aug arguments
* README and documentation overhaul. See initial (WIP) documentation at https://rwightman.github.io/pytorch-image-models/
* adamp and sgdp optimizers added by [@hellbell](https://github.com/hellbell)

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

Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arxiv papers, etc in the README, documentation, and code docstrings. Please let me know if I missed anything.

## Models

All model architecture families include variants with pretrained weights. The are variants without any weights. Help training new or better weights is always appreciated. Here are some example [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) to get you started.

A full version of the list below with source links can be found in the [documentation](https://rwightman.github.io/pytorch-image-models/models/).

* CspNet (Cross-Stage Partial Networks) - https://arxiv.org/abs/1911.11929
* DenseNet - https://arxiv.org/abs/1608.06993
* DLA - https://arxiv.org/abs/1707.06484
* DPN (Dual-Path Network) - https://arxiv.org/abs/1707.01629
* EfficientNet (MBConvNet Family)
    * EfficientNet NoisyStudent (B0-B7, L2) - https://arxiv.org/abs/1911.04252
    * EfficientNet AdvProp (B0-B8) - https://arxiv.org/abs/1911.09665
    * EfficientNet (B0-B7) - https://arxiv.org/abs/1905.11946
    * EfficientNet-EdgeTPU (S, M, L) - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
    * FBNet-C - https://arxiv.org/abs/1812.03443
    * MixNet - https://arxiv.org/abs/1907.09595
    * MNASNet B1, A1 (Squeeze-Excite), and Small - https://arxiv.org/abs/1807.11626
    * MobileNet-V2 - https://arxiv.org/abs/1801.04381
    * Single-Path NAS - https://arxiv.org/abs/1904.02877
* HRNet - https://arxiv.org/abs/1908.07919
* Inception-V3 - https://arxiv.org/abs/1512.00567
* Inception-ResNet-V2 and Inception-V4 - https://arxiv.org/abs/1602.07261
* MobileNet-V3 (MBConvNet w/ Efficient Head) - https://arxiv.org/abs/1905.02244
* NASNet-A - https://arxiv.org/abs/1707.07012
* PNasNet - https://arxiv.org/abs/1712.00559
* RegNet - https://arxiv.org/abs/2003.13678
* ResNet/ResNeXt
    * ResNet (v1b/v1.5) - https://arxiv.org/abs/1512.03385
    * ResNeXt - https://arxiv.org/abs/1611.05431
    * 'Bag of Tricks' / Gluon C, D, E, S variations - https://arxiv.org/abs/1812.01187
    * Weakly-supervised (WSL) Instagram pretrained / ImageNet tuned ResNeXt101 - https://arxiv.org/abs/1805.00932
    * Semi-supervised (SSL) / Semi-weakly Supervised (SWSL) ResNet/ResNeXts - https://arxiv.org/abs/1905.00546
    * ECA-Net (ECAResNet) - https://arxiv.org/abs/1910.03151v4
    * Squeeze-and-Excitation Networks (SEResNet) - https://arxiv.org/abs/1709.01507
* Res2Net - https://arxiv.org/abs/1904.01169
* ResNeSt - https://arxiv.org/abs/2004.08955
* ReXNet - https://arxiv.org/abs/2007.00992
* SelecSLS - https://arxiv.org/abs/1907.00837
* Selective Kernel Networks - https://arxiv.org/abs/1903.06586
* TResNet - https://arxiv.org/abs/2003.13630
* Vision Transformer - https://arxiv.org/abs/2010.11929
* VovNet V2 and V1 - https://arxiv.org/abs/1911.06667
* Xception - https://arxiv.org/abs/1610.02357
* Xception (Modified Aligned, Gluon) - https://arxiv.org/abs/1802.02611
* Xception (Modified Aligned, TF) - https://arxiv.org/abs/1802.02611

## Features

Several (less common) features that I often utilize in my projects are included. Many of their additions are the reason why I maintain my own set of models, instead of using others' via PIP:

* All models have a common default configuration interface and API for
    * accessing/changing the classifier - `get_classifier` and `reset_classifier`
    * doing a forward pass on just the features - `forward_features` (see [documentation](https://rwightman.github.io/pytorch-image-models/feature_extraction/))
    * these makes it easy to write consistent network wrappers that work with any of the models
* All models support multi-scale feature map extraction (feature pyramids) via create_model (see [documentation](https://rwightman.github.io/pytorch-image-models/feature_extraction/))
    * `create_model(name, features_only=True, out_indices=..., output_stride=...)`
    * `out_indices` creation arg specifies which feature maps to return, these indices are 0 based and generally correspond to the `C(i + 1)` feature level.
    * `output_stride` creation arg controls output stride of the network by using dilated convolutions. Most networks are stride 32 by default. Not all networks support this.
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
    * `adafactor` adapted from [FAIRSeq impl](https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py) (https://arxiv.org/abs/1804.04235)
    * `adahessian` by [David Samuel](https://github.com/davda54/ada-hessian) (https://arxiv.org/abs/2006.00719)
* Random Erasing from [Zhun Zhong](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py)  (https://arxiv.org/abs/1708.04896)
* Mixup (https://arxiv.org/abs/1710.09412)
* CutMix (https://arxiv.org/abs/1905.04899)
* AutoAugment (https://arxiv.org/abs/1805.09501) and RandAugment (https://arxiv.org/abs/1909.13719) ImageNet configurations modeled after impl for EfficientNet training (https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py)
* AugMix w/ JSD loss (https://arxiv.org/abs/1912.02781), JSD w/ clean + augmented mixing support works with AutoAugment and RandAugment as well
* SplitBachNorm - allows splitting batch norm layers between clean and augmented (auxiliary batch norm) data
* DropPath aka "Stochastic Depth" (https://arxiv.org/abs/1603.09382) 
* DropBlock (https://arxiv.org/abs/1810.12890)
* Efficient Channel Attention - ECA (https://arxiv.org/abs/1910.03151)
* Blur Pooling (https://arxiv.org/abs/1904.11486)
* Space-to-Depth by [mrT23](https://github.com/mrT23/TResNet/blob/master/src/models/tresnet/layers/space_to_depth.py) (https://arxiv.org/abs/1801.04590) -- original paper?

## Results

Model validation results can be found in the [documentation](https://rwightman.github.io/pytorch-image-models/results/) and in the [results tables](results/README.md)

## Getting Started

See the [documentation](https://rwightman.github.io/pytorch-image-models/)

## Train, Validation, Inference Scripts

The root folder of the repository contains reference train, validation, and inference scripts that work with the included models and other features of this repository. They are adaptable for other datasets and use cases with a little hacking. See [documentation](https://rwightman.github.io/pytorch-image-models/scripts/) for some basics and [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) for some train examples that produce SOTA ImageNet results.

## Awesome PyTorch Resources

One of the greatest assets of PyTorch is the community and their contributions. A few of my favourite resources that pair well with the models and componenets here are listed below.

### Training / Frameworks
* PyTorch Lightning - https://github.com/PyTorchLightning/pytorch-lightning
* fastai - https://github.com/fastai/fastai

### Computer Vision / Image Augmentation
* Albumentations - https://github.com/albumentations-team/albumentations
* Kornia - https://github.com/kornia/kornia

### Metric Learning
* PyTorch Metric Learning - https://github.com/KevinMusgrave/pytorch-metric-learning

### Object Detection, Instance and Semantic Segmentation
* Detectron2 - https://github.com/facebookresearch/detectron2
* Segmentation Models (Semantic) - https://github.com/qubvel/segmentation_models.pytorch/issues
* EfficientDet (Obj Det, Semantic soon) - https://github.com/rwightman/efficientdet-pytorch

## Licenses

### Code
The code here is licensed Apache 2.0. I've taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc. I've made an effort to avoid any GPL / LGPL conflicts. That said, it is your responsibility to ensure you comply with license here and conditions of any dependent licenses. Where applicable, I've linked the sources/references for various components in docstrings. If you think I've missed anything please create an issue.

### Pretrained Weights
So far all of the pretrained weights available here are pretrained on ImageNet with a select few that have some additional pretraining (see extra note below). ImageNet was released for non-commercial research purposes only (http://www.image-net.org/download-faq). It's not clear what the implications of that are for the use of pretrained weights from that dataset. Any models I have trained with ImageNet are done for research purposes and one should assume that the original dataset license applies to the weights. It's best to seek legal advice if you intend to use the pretrained weights in a commercial product.

#### Pretrained on more than ImageNet
Several weights included or references here were pretrained with proprietary datasets that I do not have access to. These include the Facebook WSL, SSL, SWSL ResNe(Xt) and the Google Noisy Student EfficientNet models. The Facebook models have an explicit non-commercial license (CC-BY-NC 4.0, https://github.com/facebookresearch/semi-supervised-ImageNet1K-models, https://github.com/facebookresearch/WSL-Images). The Google models do not appear to have any restriction beyond the Apache 2.0 license (and ImageNet concerns). In either case, you should contact Facebook or Google with any questions.

