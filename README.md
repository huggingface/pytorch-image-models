# PyTorch Image Models
- [Sponsors](#sponsors)
- [What's New](#whats-new)
- [Introduction](#introduction)
- [Models](#models)
- [Features](#features)
- [Results](#results)
- [Getting Started (Documentation)](#getting-started-documentation)
- [Train, Validation, Inference Scripts](#train-validation-inference-scripts)
- [Awesome PyTorch Resources](#awesome-pytorch-resources)
- [Licenses](#licenses)
- [Citing](#citing)

## Sponsors

A big thank you to my [GitHub Sponsors](https://github.com/sponsors/rwightman) for their support!

In addition to the sponsors at the link above, I've received hardware and/or cloud resources from
* Nvidia (https://www.nvidia.com/en-us/)
* TFRC (https://www.tensorflow.org/tfrc)

I'm fortunate to be able to dedicate significant time and money of my own supporting this and other open source projects. However, as the projects increase in scope, outside support is needed to continue with the current trajectory of hardware, infrastructure, and electricty costs.

## What's New

### April 1, 2021
* Add snazzy `benchmark.py` script for bulk `timm` model benchmarking of train and/or inference
* Add Pooling-based Vision Transformer (PiT) models (from https://github.com/naver-ai/pit)
  * Merged distilled variant into main for torchscript compatibility
  * Some `timm` cleanup/style tweaks and weights have hub download support
* Cleanup Vision Transformer (ViT) models
  * Merge distilled (DeiT) model into main so that torchscript can work
  * Support updated weight init (defaults to old still) that closer matches original JAX impl (possibly better training from scratch)
  * Separate hybrid model defs into different file and add several new model defs to fiddle with, support patch_size != 1 for hybrids
  * Fix fine-tuning num_class changes (PiT and ViT) and pos_embed resizing (Vit) with distilled variants
  * nn.Sequential for block stack (does not break downstream compat)
* TnT (Transformer-in-Transformer) models contributed by author (from https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT)
* Add RegNetY-160 weights from DeiT teacher model
* Add new NFNet-L0 w/ SE attn (rename `nfnet_l0b`->`nfnet_l0`) weights 82.75 top-1 @ 288x288
* Some fixes/improvements for TFDS dataset wrapper

### March 17, 2021
* Add new ECA-NFNet-L0 (rename `nfnet_l0c`->`eca_nfnet_l0`) weights trained by myself.
  * 82.6 top-1 @ 288x288, 82.8 @ 320x320, trained at 224x224
  * Uses SiLU activation, approx 2x faster than `dm_nfnet_f0` and 50% faster than `nfnet_f0s` w/ 1/3 param count
* Integrate [Hugging Face model hub](https://huggingface.co/models) into timm create_model and default_cfg handling for pretrained weight and config sharing (more on this soon!)
* Merge HardCoRe NAS models contributed by https://github.com/yoniaflalo
* Merge PyTorch trained EfficientNet-EL and pruned ES/EL variants contributed by [DeGirum](https://github.com/DeGirum)


### March 7, 2021
* First 0.4.x PyPi release w/ NFNets (& related), ByoB (GPU-Efficient, RepVGG, etc).
* Change feature extraction for pre-activation nets (NFNets, ResNetV2) to return features before activation.
* Tested with PyTorch 1.8 release. Updated CI to use 1.8.
* Benchmarked several arch on RTX 3090, Titan RTX, and V100 across 1.7.1, 1.8, NGC 20.12, and 21.02. Some interesting performance variations to take note of https://gist.github.com/rwightman/bb59f9e245162cee0e38bd66bd8cd77f

### Feb 18, 2021
* Add pretrained weights and model variants for NFNet-F* models from [DeepMind Haiku impl](https://github.com/deepmind/deepmind-research/tree/master/nfnets).
  * Models are prefixed with `dm_`. They require SAME padding conv, skipinit enabled, and activation gains applied in act fn.
  * These models are big, expect to run out of GPU memory. With the GELU activiation + other options, they are roughly 1/2 the inference speed of my SiLU PyTorch optimized `s` variants.
  * Original model results are based on pre-processing that is not the same as all other models so you'll see different results in the results csv (once updated).
  * Matching the original pre-processing as closely as possible I get these results:
    * `dm_nfnet_f6` - 86.352
    * `dm_nfnet_f5` - 86.100
    * `dm_nfnet_f4` - 85.834
    * `dm_nfnet_f3` - 85.676
    * `dm_nfnet_f2` - 85.178
    * `dm_nfnet_f1` - 84.696
    * `dm_nfnet_f0` - 83.464

### Feb 16, 2021
* Add Adaptive Gradient Clipping (AGC) as per https://arxiv.org/abs/2102.06171. Integrated w/ PyTorch gradient clipping via mode arg that defaults to prev 'norm' mode. For backward arg compat, clip-grad arg must be specified to enable when using train.py.
  * AGC w/ default clipping factor `--clip-grad .01 --clip-mode agc`
  * PyTorch global norm of 1.0 (old behaviour, always norm), `--clip-grad 1.0`
  * PyTorch value clipping of 10, `--clip-grad 10. --clip-mode value`
  * AGC performance is definitely sensitive to the clipping factor. More experimentation needed to determine good values for smaller batch sizes and optimizers besides those in paper. So far I've found .001-.005 is necessary for stable RMSProp training w/ NFNet/NF-ResNet.

### Feb 12, 2021
* Update Normalization-Free nets to include new NFNet-F (https://arxiv.org/abs/2102.06171) model defs

### Feb 10, 2021
* First Normalization-Free model training experiments done,
  * nf_resnet50 - 80.68 top-1 @ 288x288, 80.31 @ 256x256
  * nf_regnet_b1 - 79.30 @ 288x288, 78.75 @ 256x256
* More model archs, incl a flexible ByobNet backbone ('Bring-your-own-blocks')
  * GPU-Efficient-Networks (https://github.com/idstcv/GPU-Efficient-Networks), impl in `byobnet.py`
  * RepVGG (https://github.com/DingXiaoH/RepVGG), impl in `byobnet.py`
  * classic VGG (from torchvision, impl in `vgg.py`)
* Refinements to normalizer layer arg handling and normalizer+act layer handling in some models
* Default AMP mode changed to native PyTorch AMP instead of APEX. Issues not being fixed with APEX. Native works with `--channels-last` and `--torchscript` model training, APEX does not.
* Fix a few bugs introduced since last pypi release

### Feb 8, 2021
* Add several ResNet weights with ECA attention. 26t & 50t trained @ 256, test @ 320. 269d train @ 256, fine-tune @320, test @ 352.
  * `ecaresnet26t` - 79.88 top-1 @ 320x320, 79.08 @ 256x256
  * `ecaresnet50t` - 82.35 top-1 @ 320x320, 81.52 @ 256x256
  * `ecaresnet269d` - 84.93 top-1 @ 352x352, 84.87 @ 320x320
* Remove separate tiered (`t`) vs tiered_narrow (`tn`) ResNet model defs, all `tn` changed to `t` and `t` models removed (`seresnext26t_32x4d` only model w/ weights that was removed).
* Support model default_cfgs with separate train vs test resolution `test_input_size` and remove extra `_320` suffix ResNet model defs that were just for test.

### Jan 30, 2021
* Add initial "Normalization Free" NF-RegNet-B* and NF-ResNet model definitions based on [paper](https://arxiv.org/abs/2101.08692)

### Jan 25, 2021
* Add ResNetV2 Big Transfer (BiT) models w/ ImageNet-1k and 21k weights from https://github.com/google-research/big_transfer
* Add official R50+ViT-B/16 hybrid models + weights from https://github.com/google-research/vision_transformer
* ImageNet-21k ViT weights are added w/ model defs and representation layer (pre logits) support
  * NOTE: ImageNet-21k classifier heads were zero'd in original weights, they are only useful for transfer learning
* Add model defs and weights for DeiT Vision Transformer models from https://github.com/facebookresearch/deit
* Refactor dataset classes into ImageDataset/IterableImageDataset + dataset specific parser classes
* Add Tensorflow-Datasets (TFDS) wrapper to allow use of TFDS image classification sets with train script
  * Ex: `train.py /data/tfds --dataset tfds/oxford_iiit_pet --val-split test --model resnet50 -b 256 --amp --num-classes 37 --opt adamw --lr 3e-4 --weight-decay .001 --pretrained -j 2`
* Add improved .tar dataset parser that reads images from .tar, folder of .tar files, or .tar within .tar
  * Run validation on full ImageNet-21k directly from tar w/ BiT model: `validate.py /data/fall11_whole.tar --model resnetv2_50x1_bitm_in21k --amp`
* Models in this update should be stable w/ possible exception of ViT/BiT, possibility of some regressions with train/val scripts and dataset handling

### Jan 3, 2021
* Add SE-ResNet-152D weights
  * 256x256 val, 0.94 crop top-1 - 83.75
  * 320x320 val, 1.0 crop - 84.36
* Update [results files](results/)

### Dec 18, 2020
* Add ResNet-101D, ResNet-152D, and ResNet-200D weights trained @ 256x256
  * 256x256 val, 0.94 crop (top-1) - 101D (82.33), 152D (83.08), 200D (83.25)
  * 288x288 val, 1.0 crop - 101D (82.64), 152D (83.48), 200D (83.76)
  * 320x320 val, 1.0 crop - 101D (83.00), 152D (83.66), 200D (84.01)

### Dec 7, 2020
* Simplify EMA module (ModelEmaV2), compatible with fully torchscripted models
* Misc fixes for SiLU ONNX export, default_cfg missing from Feature extraction models, Linear layer w/ AMP + torchscript
* PyPi release @ 0.3.2 (needed by EfficientDet)

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


## Introduction

Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arxiv papers, etc in the README, documentation, and code docstrings. Please let me know if I missed anything.

## Models

All model architecture families include variants with pretrained weights. There are specific model variants without any weights, it is NOT a bug. Help training new or better weights is always appreciated. Here are some example [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) to get you started.

A full version of the list below with source links can be found in the [documentation](https://rwightman.github.io/pytorch-image-models/models/).

* Big Transfer ResNetV2 (BiT) - https://arxiv.org/abs/1912.11370
* CspNet (Cross-Stage Partial Networks) - https://arxiv.org/abs/1911.11929
* DeiT (Vision Transformer) - https://arxiv.org/abs/2012.12877
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
* GPU-Efficient Networks - https://arxiv.org/abs/2006.14090
* HardCoRe-NAS - https://arxiv.org/abs/2102.11646
* HRNet - https://arxiv.org/abs/1908.07919
* Inception-V3 - https://arxiv.org/abs/1512.00567
* Inception-ResNet-V2 and Inception-V4 - https://arxiv.org/abs/1602.07261
* MobileNet-V3 (MBConvNet w/ Efficient Head) - https://arxiv.org/abs/1905.02244
* NASNet-A - https://arxiv.org/abs/1707.07012
* NFNet-F - https://arxiv.org/abs/2102.06171
* NF-RegNet / NF-ResNet - https://arxiv.org/abs/2101.08692
* PNasNet - https://arxiv.org/abs/1712.00559
* Pooling-based Vision Transformer (PiT) - https://arxiv.org/abs/2103.16302
* RegNet - https://arxiv.org/abs/2003.13678
* RepVGG - https://arxiv.org/abs/2101.03697
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
* Transformer-iN-Transformer (TNT) - https://arxiv.org/abs/2103.00112
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
* Adaptive Gradient Clipping (https://arxiv.org/abs/2102.06171, https://github.com/deepmind/deepmind-research/tree/master/nfnets)

## Results

Model validation results can be found in the [documentation](https://rwightman.github.io/pytorch-image-models/results/) and in the [results tables](results/README.md)

## Getting Started (Documentation)

My current [documentation](https://rwightman.github.io/pytorch-image-models/) for `timm` covers the basics.

[timmdocs](https://fastai.github.io/timmdocs/) is quickly becoming a much more comprehensive set of documentation for `timm`. A big thanks to [Aman Arora](https://github.com/amaarora) for his efforts creating timmdocs.

[paperswithcode](https://paperswithcode.com/lib/timm) is a good resource for browsing the models within `timm`.

## Train, Validation, Inference Scripts

The root folder of the repository contains reference train, validation, and inference scripts that work with the included models and other features of this repository. They are adaptable for other datasets and use cases with a little hacking. See [documentation](https://rwightman.github.io/pytorch-image-models/scripts/) for some basics and [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) for some train examples that produce SOTA ImageNet results.

## Awesome PyTorch Resources

One of the greatest assets of PyTorch is the community and their contributions. A few of my favourite resources that pair well with the models and componenets here are listed below.

### Object Detection, Instance and Semantic Segmentation
* Detectron2 - https://github.com/facebookresearch/detectron2
* Segmentation Models (Semantic) - https://github.com/qubvel/segmentation_models.pytorch
* EfficientDet (Obj Det, Semantic soon) - https://github.com/rwightman/efficientdet-pytorch

### Computer Vision / Image Augmentation
* Albumentations - https://github.com/albumentations-team/albumentations
* Kornia - https://github.com/kornia/kornia

### Knowledge Distillation
* RepDistiller - https://github.com/HobbitLong/RepDistiller
* torchdistill - https://github.com/yoshitomo-matsubara/torchdistill

### Metric Learning
* PyTorch Metric Learning - https://github.com/KevinMusgrave/pytorch-metric-learning

### Training / Frameworks
* fastai - https://github.com/fastai/fastai

## Licenses

### Code
The code here is licensed Apache 2.0. I've taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc. I've made an effort to avoid any GPL / LGPL conflicts. That said, it is your responsibility to ensure you comply with license here and conditions of any dependent licenses. Where applicable, I've linked the sources/references for various components in docstrings. If you think I've missed anything please create an issue.

### Pretrained Weights
So far all of the pretrained weights available here are pretrained on ImageNet with a select few that have some additional pretraining (see extra note below). ImageNet was released for non-commercial research purposes only (http://www.image-net.org/download-faq). It's not clear what the implications of that are for the use of pretrained weights from that dataset. Any models I have trained with ImageNet are done for research purposes and one should assume that the original dataset license applies to the weights. It's best to seek legal advice if you intend to use the pretrained weights in a commercial product.

#### Pretrained on more than ImageNet
Several weights included or references here were pretrained with proprietary datasets that I do not have access to. These include the Facebook WSL, SSL, SWSL ResNe(Xt) and the Google Noisy Student EfficientNet models. The Facebook models have an explicit non-commercial license (CC-BY-NC 4.0, https://github.com/facebookresearch/semi-supervised-ImageNet1K-models, https://github.com/facebookresearch/WSL-Images). The Google models do not appear to have any restriction beyond the Apache 2.0 license (and ImageNet concerns). In either case, you should contact Facebook or Google with any questions.

## Citing

### BibTeX

```
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

### Latest DOI

[![DOI](https://zenodo.org/badge/168799526.svg)](https://zenodo.org/badge/latestdoi/168799526)
