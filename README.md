# PyTorch Image Models, etc

## What's New

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

### Feb 29, 2020
* New MobileNet-V3 Large weights trained from stratch with this code to 75.77% top-1
* IMPORTANT CHANGE - default weight init changed for all MobilenetV3 / EfficientNet / related models
  * overall results similar to a bit better training from scratch on a few smaller models tried
  * performance early in training seems consistently improved but less difference by end
  * set `fix_group_fanout=False` in `_init_weight_goog` fn if you need to reproducte past behaviour
* Experimental LR noise feature added applies a random perturbation to LR each epoch in specified range of training

### Feb 18, 2020
* Big refactor of model layers and addition of several attention mechanisms. Several additions motivated by 'Compounding the Performance Improvements...' (https://arxiv.org/abs/2001.06268):
  * Move layer/module impl into `layers` subfolder/module of `models` and organize in a more granular fashion
  * ResNet downsample paths now properly support dilation (output stride != 32) for avg_pool ('D' variant) and 3x3 (SENets) networks
  * Add Selective Kernel Nets on top of ResNet base, pretrained weights
    * skresnet18 - 73% top-1
    * skresnet34 - 76.9% top-1 
    * skresnext50_32x4d (equiv to SKNet50) - 80.2% top-1
  * ECA and CECA (circular padding) attention layer contributed by [Chris Ha](https://github.com/VRandme)
  * CBAM attention experiment (not the best results so far, may remove)
  * Attention factory to allow dynamically selecting one of SE, ECA, CBAM in the `.se` position for all ResNets
  * Add DropBlock and DropPath (formerly DropConnect for EfficientNet/MobileNetv3) support to all ResNet variants
* Full dataset results updated that incl NoisyStudent weights and 2 of the 3 SK weights

### Feb 12, 2020
* Add EfficientNet-L2 and B0-B7 NoisyStudent weights ported from [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## Introduction 

For each competition, personal, or freelance project involving images + Convolution Neural Networks, I build on top of an evolving collection of code and models. This repo contains a (somewhat) cleaned up and paired down iteration of that code. Hopefully it'll be of use to others.

The work of many others is present here. I've tried to make sure all source material is acknowledged:
* Training/validation scripts evolved from early versions of the [PyTorch Imagenet Examples](https://github.com/pytorch/examples)
* CUDA specific performance enhancements have been pulled from [NVIDIA's APEX Examples](https://github.com/NVIDIA/apex/tree/master/examples)
* LR scheduler ideas from [AllenNLP](https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers), [FAIRseq](https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler), and SGDR: Stochastic Gradient Descent with Warm Restarts (https://arxiv.org/abs/1608.03983)
* Random Erasing from [Zhun Zhong](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py)  (https://arxiv.org/abs/1708.04896)
* Optimizers:
    * RAdam by [Liyuan Liu](https://github.com/LiyuanLucasLiu/RAdam) (https://arxiv.org/abs/1908.03265)
    * NovoGrad by [Masashi Kimura](https://github.com/convergence-lab/novograd) (https://arxiv.org/abs/1905.11286)
    * Lookahead adapted from impl by [Liam](https://github.com/alphadl/lookahead.pytorch) (https://arxiv.org/abs/1907.08610)

## Models

I've included a few of my favourite models, but this is not an exhaustive collection. You can't do better than [Cadene's](https://github.com/Cadene/pretrained-models.pytorch) collection in that regard. Most models do have pretrained weights from their respective sources or original authors.

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
    * Original variant from [Cadene](https://github.com/Cadene/pretrained-models.pytorch)
    * MXNet Gluon 'modified aligned' Xception-65 and 71 models from [Gluon ModelZoo](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo)
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
    * code from https://github.com/HRNet/HRNet-Image-Classification, paper https://arxiv.org/abs/1908.07919
* SelecSLS
    * code from https://github.com/mehtadushy/SelecSLS-Pytorch, paper https://arxiv.org/abs/1907.00837
* TResNet
    * code from https://github.com/mrT23/TResNet, paper https://arxiv.org/abs/2003.13630
* RegNet
    * paper `Designing Network Design Spaces` - https://arxiv.org/abs/2003.13678
    * reference code at https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
* VovNet V2 (with V1 support)
    * paper `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    * reference code at https://github.com/youngwanLEE/vovnet-detectron2

Use the  `--model` arg to specify model for train, validation, inference scripts. Match the all lowercase
creation fn for the model you'd like.

## Features
Several (less common) features that I often utilize in my projects are included. Many of their additions are the reason why I maintain my own set of models, instead of using others' via PIP:
* All models have a common default configuration interface and API for
    * accessing/changing the classifier - `get_classifier` and `reset_classifier`
    * doing a forward pass on just the features - `forward_features`
    * these makes it easy to write consistent network wrappers that work with any of the models
* All models have a consistent pretrained weight loader that adapts last linear if necessary, and from 3 to 1 channel input if desired
* The train script works in several process/GPU modes:
    * NVIDIA DDP w/ a single GPU per process, multiple processes with APEX present (AMP mixed-precision optional)
    * PyTorch DistributedDataParallel w/ multi-gpu, single process (AMP disabled as it crashes when enabled)
    * PyTorch w/ single GPU single process (AMP optional)
* A dynamic global pool implementation that allows selecting from average pooling, max pooling, average + max, or concat([average, max]) at model creation. All global pooling is adaptive average by default and compatible with pretrained weights.
* A 'Test Time Pool' wrapper that can wrap any of the included models and usually provide improved performance doing inference with input images larger than the training size. Idea adapted from original DPN implementation when I ported (https://github.com/cypw/DPNs)
* Training schedules and techniques that provide competitive results (Cosine LR, Random Erasing, Label Smoothing, etc)
* Mixup (as in https://arxiv.org/abs/1710.09412) - currently implementing/testing
* An inference script that dumps output to CSV is provided as an example
* AutoAugment (https://arxiv.org/abs/1805.09501) and RandAugment (https://arxiv.org/abs/1909.13719) ImageNet configurations modeled after impl for EfficientNet training (https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py)
* AugMix w/ JSD loss (https://arxiv.org/abs/1912.02781), JSD w/ clean + augmented mixing support works with AutoAugment and RandAugment as well
* SplitBachNorm - allows splitting batch norm layers between clean and augmented (auxiliary batch norm) data
* DropBlock (https://arxiv.org/abs/1810.12890)
* Efficient Channel Attention - ECA (https://arxiv.org/abs/1910.03151)

## Results

A CSV file containing an ImageNet-1K validation results summary for all included models with pretrained weights and default configurations is located [here](results/results-all.csv)

### Self-trained Weights
I've leveraged the training scripts in this repository to train a few of the models with missing weights to good levels of performance. These numbers are all for 224x224 training and validation image sizing with the usual 87.5% validation crop.

|Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling  | Image Size |
|---|---|---|---|---|---|
| efficientnet_b3a | 81.874 (18.126) | 95.840 (4.160) | 12.23M | bicubic | 320 (1.0 crop) |
| efficientnet_b3 | 81.498 (18.502) | 95.718 (4.282) | 12.23M | bicubic | 300 |
| skresnext50d_32x4d | 81.278 (18.722) | 95.366 (4.634) | 27.5M | bicubic | 288 (1.0 crop) |
| efficientnet_b2a | 80.608 (19.392) | 95.310 (4.690) | 9.11M | bicubic | 288 (1.0 crop) |
| mixnet_xl | 80.478 (19.522) | 94.932 (5.068) | 11.90M | bicubic | 224 |
| efficientnet_b2 | 80.402 (19.598) | 95.076 (4.924) | 9.11M | bicubic | 260 |
| skresnext50d_32x4d | 80.156 (19.844) | 94.642 (5.358) | 27.5M | bicubic | 224 |
| resnext50_32x4d | 79.762 (20.238) | 94.600 (5.400) | 25M | bicubic | 224 |
| resnext50d_32x4d | 79.674 (20.326) | 94.868 (5.132) | 25.1M | bicubic | 224 |
| ese_vovnet39b | 79.320 (20.680) | 94.710 (5.290) | 24.6M | bicubic | 224 |
| resnetblur50 | 79.290 (20.710) | 94.632 (5.368) | 25.6M | bicubic | 224 |
| resnet50 | 79.038 (20.962) | 94.390 (5.610) | 25.6M | bicubic | 224 |
| mixnet_l | 78.976 (21.024 | 94.184 (5.816) | 7.33M | bicubic | 224 |
| efficientnet_b1 | 78.692 (21.308) | 94.086 (5.914) | 7.79M | bicubic | 240 |
| efficientnet_es | 78.066 (21.934) | 93.926 (6.074) | 5.44M | bicubic | 224 |
| seresnext26t_32x4d | 77.998 (22.002) | 93.708 (6.292) | 16.8M | bicubic | 224 |
| seresnext26tn_32x4d | 77.986 (22.014) | 93.746 (6.254) | 16.8M | bicubic | 224 |
| efficientnet_b0 | 77.698 (22.302) | 93.532 (6.468) | 5.29M | bicubic | 224 |
| seresnext26d_32x4d | 77.602 (22.398) | 93.608 (6.392) | 16.8M | bicubic | 224 |
| mobilenetv2_120d | 77.294 (22.706 | 93.502 (6.498) | 5.8M | bicubic | 224 |
| mixnet_m | 77.256 (22.744) | 93.418 (6.582) | 5.01M | bicubic | 224 |
| seresnext26_32x4d | 77.104 (22.896) | 93.316 (6.684) | 16.8M | bicubic | 224 |
| skresnet34 | 76.912 (23.088) | 93.322 (6.678) | 22.2M | bicubic | 224 |
| ese_vovnet19b_dw | 76.798 (23.202) | 93.268 (6.732) | 6.5M | bicubic | 224 |
| resnet26d | 76.68 (23.32) | 93.166 (6.834) | 16M | bicubic | 224 |
| densenetblur121d | 76.576 (23.424) | 93.190 (6.810) | 8.0M | bicubic | 224 |
| mobilenetv2_140 | 76.524 (23.476) | 92.990 (7.010) | 6.1M | bicubic | 224 |
| mixnet_s | 75.988 (24.012) | 92.794 (7.206) | 4.13M | bicubic | 224 |
| mobilenetv3_large_100 | 75.766 (24.234) | 92.542 (7.458) | 5.5M | bicubic | 224 |
| mobilenetv3_rw | 75.634 (24.366) | 92.708 (7.292) | 5.5M | bicubic | 224 |
| mnasnet_a1 | 75.448 (24.552) | 92.604 (7.396) | 3.89M | bicubic | 224 |
| resnet26 | 75.292 (24.708) | 92.57 (7.43) | 16M | bicubic | 224 |
| fbnetc_100 | 75.124 (24.876) | 92.386 (7.614) | 5.6M | bilinear | 224 |
| resnet34 | 75.110 (24.890) | 92.284 (7.716) | 22M | bilinear | 224 |
| mobilenetv2_110d | 75.052 (24.948) | 92.180 (7.820) | 4.5M | bicubic | 224 |
| seresnet34 | 74.808 (25.192) | 92.124 (7.876) | 22M | bilinear | 224 |
| mnasnet_b1 | 74.658 (25.342) | 92.114 (7.886) | 4.38M | bicubic | 224 |
| spnasnet_100 | 74.084 (25.916)  | 91.818 (8.182) | 4.42M | bilinear | 224 |
| skresnet18 | 73.038 (26.962) | 91.168 (8.832) | 11.9M | bicubic | 224 |
| mobilenetv2_100 | 72.978 (27.022) | 91.016 (8.984) | 3.5M | bicubic | 224 |
| seresnet18 | 71.742 (28.258) | 90.334 (9.666) | 11.8M | bicubic | 224 |

### Ported Weights
For the models below, the model code and weight porting from Tensorflow or MXNet Gluon to Pytorch was done by myself. There are weights/models ported by others included in this repository, they are not listed below.

| Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling | Image Size |
|---|---|---|---|---|---|
| tf_efficientnet_l2_ns *tfp | 88.352 (11.648) | 98.652 (1.348) | 480 | bicubic | 800 |
| tf_efficientnet_l2_ns      | TBD | TBD | 480 | bicubic | 800 |
| tf_efficientnet_l2_ns_475      | 88.234 (11.766) | 98.546 (1.454)f | 480 | bicubic | 475 |
| tf_efficientnet_l2_ns_475 *tfp | 88.172 (11.828) | 98.566 (1.434) | 480 | bicubic | 475 |
| tf_efficientnet_b7_ns *tfp | 86.844 (13.156) | 98.084 (1.916) | 66.35 | bicubic | 600 |
| tf_efficientnet_b7_ns      | 86.840 (13.160) | 98.094 (1.906) | 66.35 | bicubic | 600 |
| tf_efficientnet_b6_ns      | 86.452 (13.548) | 97.882 (2.118) | 43.04 | bicubic | 528 |
| tf_efficientnet_b6_ns *tfp | 86.444 (13.556) | 97.880 (2.120) | 43.04 | bicubic | 528 |
| tf_efficientnet_b5_ns *tfp | 86.064 (13.936) | 97.746 (2.254) | 30.39 | bicubic | 456 |
| tf_efficientnet_b5_ns      | 86.088 (13.912) | 97.752 (2.248) | 30.39 | bicubic | 456 |
| tf_efficientnet_b8_ap *tfp | 85.436 (14.564) | 97.272 (2.728) | 87.4 | bicubic | 672 |
| tf_efficientnet_b8 *tfp    | 85.384 (14.616) | 97.394 (2.606) | 87.4 | bicubic | 672 |
| tf_efficientnet_b8         | 85.370 (14.630) | 97.390 (2.610) | 87.4 | bicubic | 672 |
| tf_efficientnet_b8_ap      | 85.368 (14.632) | 97.294 (2.706) | 87.4 | bicubic | 672 |
| tf_efficientnet_b4_ns *tfp | 85.298 (14.702) | 97.504 (2.496) | 19.34 | bicubic | 380 |
| tf_efficientnet_b4_ns      | 85.162 (14.838) | 97.470 (2.530) | 19.34 | bicubic | 380 |
| tf_efficientnet_b7_ap *tfp | 85.154 (14.846) | 97.244 (2.756) | 66.35 | bicubic | 600 |
| tf_efficientnet_b7_ap      | 85.118 (14.882) | 97.252 (2.748) | 66.35 | bicubic | 600 |
| tf_efficientnet_b7 *tfp    | 84.940 (15.060) | 97.214 (2.786) | 66.35 | bicubic | 600 |
| tf_efficientnet_b7         | 84.932 (15.068) | 97.208 (2.792) | 66.35 | bicubic | 600 |
| tf_efficientnet_b6_ap      | 84.786 (15.214) | 97.138 (2.862) | 43.04 | bicubic | 528 |
| tf_efficientnet_b6_ap *tfp | 84.760 (15.240) | 97.124 (2.876) | 43.04 | bicubic | 528 |
| tf_efficientnet_b5_ap *tfp | 84.276 (15.724) | 96.932 (3.068) | 30.39 | bicubic | 456 |
| tf_efficientnet_b5_ap      | 84.254 (15.746) | 96.976 (3.024) | 30.39 | bicubic | 456 |
| tf_efficientnet_b6 *tfp    | 84.140 (15.860) | 96.852 (3.148) | 43.04 | bicubic | 528 |
| tf_efficientnet_b6         | 84.110 (15.890) | 96.886 (3.114) | 43.04 | bicubic | 528 |
| tf_efficientnet_b3_ns *tfp | 84.054 (15.946) | 96.918 (3.082) | 12.23 | bicubic | 300 |
| tf_efficientnet_b3_ns      | 84.048 (15.952) | 96.910 (3.090) | 12.23 | bicubic | 300 |
| tf_efficientnet_b5 *tfp    | 83.822 (16.178) | 96.756 (3.244) | 30.39 | bicubic | 456 |
| tf_efficientnet_b5         | 83.812 (16.188) | 96.748 (3.252) | 30.39 | bicubic | 456 |
| tf_efficientnet_b4_ap *tfp | 83.278 (16.722) | 96.376 (3.624) | 19.34 | bicubic | 380 |
| tf_efficientnet_b4_ap      | 83.248 (16.752) | 96.388 (3.612) | 19.34 | bicubic | 380 |
| tf_efficientnet_b4         | 83.022 (16.978) | 96.300 (3.700) | 19.34 | bicubic | 380 |
| tf_efficientnet_b4 *tfp    | 82.948 (17.052) | 96.308 (3.692) | 19.34 | bicubic | 380 |
| tf_efficientnet_b2_ns *tfp | 82.436 (17.564) | 96.268 (3.732) | 9.11 | bicubic | 260 |
| tf_efficientnet_b2_ns      | 82.380 (17.620) | 96.248 (3.752) | 9.11 | bicubic | 260 |
| tf_efficientnet_b3_ap *tfp | 81.882 (18.118) | 95.662 (4.338) | 12.23 | bicubic | 300 |
| tf_efficientnet_b3_ap      | 81.828 (18.172) | 95.624 (4.376) | 12.23 | bicubic | 300 |
| tf_efficientnet_b3         | 81.636 (18.364) | 95.718 (4.282) | 12.23 | bicubic | 300 |
| tf_efficientnet_b3 *tfp    | 81.576 (18.424) | 95.662 (4.338) | 12.23 | bicubic | 300 |
| tf_efficientnet_lite4      | 81.528 (18.472) | 95.668 (4.332) | 13.00  | bilinear | 380 |
| tf_efficientnet_b1_ns *tfp | 81.514 (18.486) | 95.776 (4.224) | 7.79 | bicubic | 240 |
| tf_efficientnet_lite4 *tfp | 81.502 (18.498) | 95.676 (4.324) | 13.00  | bilinear | 380 |
| tf_efficientnet_b1_ns      | 81.388 (18.612) | 95.738 (4.262) | 7.79 | bicubic | 240 |
| gluon_senet154           | 81.224 (18.776) | 95.356 (4.644) | 115.09 | bicubic | 224 |
| gluon_resnet152_v1s      | 81.012 (18.988) | 95.416 (4.584) | 60.32  | bicubic | 224 |
| gluon_seresnext101_32x4d | 80.902 (19.098) | 95.294 (4.706) | 48.96  | bicubic | 224 |
| gluon_seresnext101_64x4d | 80.890 (19.110) | 95.304 (4.696) | 88.23  | bicubic | 224 |
| gluon_resnext101_64x4d   | 80.602 (19.398) | 94.994 (5.006) | 83.46  | bicubic | 224 |
| tf_efficientnet_el       | 80.534 (19.466) | 95.190 (4.810) | 10.59 | bicubic | 300 |
| tf_efficientnet_el *tfp  | 80.476 (19.524) | 95.200 (4.800) | 10.59 | bicubic | 300 |
| gluon_resnet152_v1d      | 80.470 (19.530) | 95.206 (4.794) | 60.21  | bicubic | 224 |
| gluon_resnet101_v1d      | 80.424 (19.576) | 95.020 (4.980) | 44.57  | bicubic | 224 |
| tf_efficientnet_b2_ap *tfp | 80.420 (19.580) | 95.040 (4.960) | 9.11 | bicubic | 260 |
| gluon_resnext101_32x4d   | 80.334 (19.666) | 94.926 (5.074) | 44.18  | bicubic | 224 |
| tf_efficientnet_b2_ap    | 80.306 (19.694) | 95.028 (4.972) | 9.11 | bicubic | 260 |
| gluon_resnet101_v1s      | 80.300 (19.700) | 95.150 (4.850) | 44.67  | bicubic | 224 |
| tf_efficientnet_b2 *tfp  | 80.188 (19.812) | 94.974 (5.026) | 9.11  | bicubic | 260 |
| tf_efficientnet_b2       | 80.086 (19.914) | 94.908 (5.092) | 9.11  | bicubic | 260 |
| gluon_resnet152_v1c      | 79.916 (20.084) | 94.842 (5.158) | 60.21  | bicubic | 224 |
| gluon_seresnext50_32x4d  | 79.912 (20.088) | 94.818 (5.182) | 27.56  | bicubic | 224 |
| tf_efficientnet_lite3       | 79.812 (20.188) | 94.914 (5.086) | 8.20  | bilinear | 300 |
| tf_efficientnet_lite3 *tfp  | 79.734 (20.266) | 94.838 (5.162) | 8.20  | bilinear | 300 |
| gluon_resnet152_v1b      | 79.692 (20.308) | 94.738 (5.262) | 60.19  | bicubic | 224 |
| gluon_xception65         | 79.604 (20.396) | 94.748 (5.252) | 39.92  | bicubic | 299 |
| gluon_resnet101_v1c      | 79.544 (20.456) | 94.586 (5.414) | 44.57  | bicubic | 224 |
| tf_efficientnet_b1_ap *tfp | 79.532 (20.468) | 94.378 (5.622) | 7.79 | bicubic | 240 |
| tf_efficientnet_cc_b1_8e *tfp | 79.464 (20.536)| 94.492 (5.508) | 39.7 | bicubic | 240 |
| gluon_resnext50_32x4d    | 79.356 (20.644) | 94.424 (5.576) | 25.03  | bicubic | 224 |
| gluon_resnet101_v1b      | 79.304 (20.696) | 94.524 (5.476) | 44.55  | bicubic | 224 |
| tf_efficientnet_cc_b1_8e | 79.298 (20.702) | 94.364 (5.636) | 39.7 | bicubic | 240 |
| tf_efficientnet_b1_ap    | 79.278 (20.722) | 94.308 (5.692) | 7.79 | bicubic | 240 |
| tf_efficientnet_b1 *tfp  | 79.172 (20.828) | 94.450 (5.550) | 7.79  | bicubic | 240 |
| gluon_resnet50_v1d       | 79.074 (20.926) | 94.476 (5.524) | 25.58  | bicubic | 224 |
| tf_efficientnet_em *tfp  | 78.958 (21.042) | 94.458 (5.542) | 6.90 | bicubic | 240 |
| tf_mixnet_l *tfp         | 78.846 (21.154) | 94.212 (5.788) | 7.33  | bilinear | 224 |
| tf_efficientnet_b1       | 78.826 (21.174) | 94.198 (5.802) | 7.79  | bicubic | 240 |
| tf_efficientnet_b0_ns *tfp | 78.806 (21.194) | 94.496 (5.504) | 5.29 | bicubic | 224 |
| gluon_inception_v3       | 78.804 (21.196) | 94.380 (5.620) | 27.16M | bicubic | 299 |
| tf_mixnet_l              | 78.770 (21.230) | 94.004 (5.996) | 7.33  | bicubic | 224 |
| tf_efficientnet_em       | 78.742 (21.258) | 94.332 (5.668) | 6.90 | bicubic | 240 |
| gluon_resnet50_v1s       | 78.712 (21.288) | 94.242 (5.758) | 25.68  | bicubic | 224 |
| tf_efficientnet_b0_ns    | 78.658 (21.342) | 94.376 (5.624) | 5.29 | bicubic | 224 |
| tf_efficientnet_cc_b0_8e *tfp | 78.314 (21.686) | 93.790 (6.210) | 24.0 | bicubic | 224 |
| gluon_resnet50_v1c       | 78.010 (21.990) | 93.988 (6.012) | 25.58  | bicubic | 224 |
| tf_efficientnet_cc_b0_8e | 77.908 (22.092) | 93.656 (6.344) | 24.0 | bicubic | 224 |
| tf_inception_v3          | 77.856 (22.144) | 93.644 (6.356) | 27.16M | bicubic | 299 |
| tf_efficientnet_cc_b0_4e *tfp | 77.746 (22.254) | 93.552 (6.448) | 13.3 | bicubic | 224 |
| tf_efficientnet_es *tfp  | 77.616 (22.384) | 93.750 (6.250) | 5.44 | bicubic | 224 |
| gluon_resnet50_v1b       | 77.578 (22.422) | 93.718 (6.282) | 25.56  | bicubic | 224 |
| adv_inception_v3         | 77.576 (22.424) | 93.724 (6.276) | 27.16M | bicubic | 299 |
| tf_efficientnet_lite2 *tfp  | 77.544 (22.456) | 93.800 (6.200) | 6.09  | bilinear | 260 |
| tf_efficientnet_lite2       | 77.460 (22.540) | 93.746 (6.254) | 6.09  | bicubic | 260 |
| tf_efficientnet_b0_ap *tfp | 77.514 (22.486) | 93.576 (6.424) | 5.29  | bicubic | 224 |
| tf_efficientnet_cc_b0_4e | 77.304 (22.696) | 93.332 (6.668) | 13.3 | bicubic | 224 |
| tf_efficientnet_es       | 77.264 (22.736) | 93.600 (6.400) | 5.44 | bicubic | 224 |
| tf_efficientnet_b0 *tfp  | 77.258 (22.742) | 93.478 (6.522) | 5.29  | bicubic | 224 |
| tf_efficientnet_b0_ap    | 77.084 (22.916) | 93.254 (6.746) | 5.29  | bicubic | 224 |
| tf_mixnet_m *tfp         | 77.072 (22.928) | 93.368 (6.632) | 5.01  | bilinear | 224 |
| tf_mixnet_m              | 76.950 (23.050) | 93.156 (6.844) | 5.01  | bicubic | 224 |
| tf_efficientnet_b0       | 76.848 (23.152) | 93.228 (6.772) | 5.29  | bicubic | 224 |
| tf_efficientnet_lite1 *tfp  | 76.764 (23.236) | 93.326 (6.674) | 5.42  | bilinear | 240 |
| tf_efficientnet_lite1       | 76.638 (23.362) | 93.232 (6.768) | 5.42  | bicubic | 240 |
| tf_mixnet_s *tfp         | 75.800 (24.200) | 92.788 (7.212) | 4.13  | bilinear | 224 |
| tf_mobilenetv3_large_100 *tfp | 75.768 (24.232) | 92.710 (7.290) | 5.48 | bilinear | 224 |
| tf_mixnet_s              | 75.648 (24.352) | 92.636 (7.364) | 4.13  | bicubic | 224 |
| tf_mobilenetv3_large_100 | 75.516 (24.484) | 92.600 (7.400) | 5.48 | bilinear | 224 |
| tf_efficientnet_lite0 *tfp  | 75.074 (24.926) | 92.314 (7.686) | 4.65  | bilinear | 224 |
| tf_efficientnet_lite0       | 74.842 (25.158) | 92.170 (7.830) | 4.65  | bicubic | 224 |
| gluon_resnet34_v1b       | 74.580 (25.420) | 91.988 (8.012) | 21.80 | bicubic | 224 |
| tf_mobilenetv3_large_075 *tfp | 73.730 (26.270) | 91.616 (8.384) | 3.99 | bilinear | 224 |
| tf_mobilenetv3_large_075 | 73.442 (26.558) | 91.352 (8.648) | 3.99 | bilinear | 224 |
| tf_mobilenetv3_large_minimal_100 *tfp | 72.678 (27.322) | 90.860 (9.140) | 3.92 | bilinear | 224 |
| tf_mobilenetv3_large_minimal_100 | 72.244 (27.756) | 90.636 (9.364) | 3.92 | bilinear | 224 |
| tf_mobilenetv3_small_100 *tfp | 67.918 (32.082) | 87.958 (12.042 | 2.54 | bilinear | 224 |
| tf_mobilenetv3_small_100 | 67.918 (32.082) | 87.662 (12.338) | 2.54 | bilinear | 224 |
| tf_mobilenetv3_small_075 *tfp | 66.142 (33.858) | 86.498 (13.502) | 2.04 | bilinear | 224 |
| tf_mobilenetv3_small_075 | 65.718 (34.282) | 86.136 (13.864) | 2.04 | bilinear | 224 |
| tf_mobilenetv3_small_minimal_100 *tfp | 63.378 (36.622) | 84.802 (15.198) | 2.04 | bilinear | 224 |
| tf_mobilenetv3_small_minimal_100 | 62.898 (37.102) | 84.230 (15.770) | 2.04 | bilinear | 224 |

Models with `*tfp` next to them were scored with `--tf-preprocessing` flag. 

The `tf_efficientnet`, `tf_mixnet` models require an equivalent for 'SAME' padding as their arch results in asymmetric padding. I've added this in the model creation wrapper, but it does come with a performance penalty. 

Sources for original weights:
* `tf_efficientnet*`: [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
* `tf_efficientnet_e*`: [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu)
* `tf_mixnet*`: [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet)
* `tf_inception*`: [Tensorflow Slim](https://github.com/tensorflow/models/tree/master/research/slim)
* `gluon_*`: [MxNet Gluon](https://gluon-cv.mxnet.io/model_zoo/classification.html)

## Training Hyperparameters

### EfficientNet-B2 with RandAugment - 80.4 top-1, 95.1 top-5
These params are for dual Titan RTX cards with NVIDIA Apex installed:

`./distributed_train.sh 2 /imagenet/ --model efficientnet_b2 -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016`

### MixNet-XL with RandAugment - 80.5 top-1, 94.9 top-5
This params are for dual Titan RTX cards with NVIDIA Apex installed:

`./distributed_train.sh 2 /imagenet/ --model mixnet_xl -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .969 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.3 --amp --lr .016 --dist-bn reduce`

### SE-ResNeXt-26-D and SE-ResNeXt-26-T
These hparams (or similar) work well for a wide range of ResNet architecture, generally a good idea to increase the epoch # as the model size increases... ie approx 180-200 for ResNe(X)t50, and 220+ for larger. Increase batch size and LR proportionally for better GPUs or with AMP enabled. These params were for 2 1080Ti cards:

`./distributed_train.sh 2 /imagenet/ --model seresnext26t_32x4d --lr 0.1 --warmup-epochs 5 --epochs 160 --weight-decay 1e-4 --sched cosine --reprob 0.4 --remode pixel -b 112`

### EfficientNet-B3 with RandAugment - 81.5 top-1, 95.7 top-5
The training of this model started with the same command line as EfficientNet-B2 w/ RA above. After almost three weeks of training the process crashed. The results weren't looking amazing so I resumed the training several times with tweaks to a few params (increase RE prob, decrease rand-aug, increase ema-decay). Nothing looked great. I ended up averaging the best checkpoints from all restarts. The result is mediocre at default res/crop but oddly performs much better with a full image test crop of 1.0. 

### EfficientNet-B0 with RandAugment - 77.7 top-1, 95.3 top-5
[Michael Klachko](https://github.com/michaelklachko) achieved these results with the command line for B2 adapted for larger batch size, with the recommended B0 dropout rate of 0.2.

`./distributed_train.sh 2 /imagenet/ --model efficientnet_b0 -b 384 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048`

### ResNet50 with JSD loss and RandAugment (clean + 2x RA augs) - 79.04 top-1, 94.39 top-5

Trained on two older 1080Ti cards, this took a while. Only slightly, non statistically better ImageNet validation result than my first good AugMix training of 78.99. However, these weights are more robust on tests with ImageNetV2, ImageNet-Sketch, etc. Unlike my first AugMix runs, I've enabled SplitBatchNorm, disabled random erasing on the clean split, and cranked up random erasing prob on the 2 augmented paths.

`./distributed_train.sh 2 /imagenet -b 64 --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce`

### EfficientNet-ES (EdgeTPU-Small) with RandAugment - 78.066 top-1, 93.926 top-5
Trained by [Andrew Lavin](https://github.com/andravin) with 8 V100 cards. Model EMA was not used, final checkpoint is the average of 8 best checkpoints during training.

`./distributed_train.sh 8 /imagenet --model efficientnet_es -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2  --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064`

### MobileNetV3-Large-100 - 75.766 top-1, 92,542 top-5

`./distributed_train.sh 2 /imagenet/ --model mobilenetv3_large_100 -b 512 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064 --lr-noise 0.42 0.9`


### ResNeXt-50 32x4d w/ RandAugment - 79.762 top-1, 94.60 top-5
These params will also work well for SE-ResNeXt-50 and SK-ResNeXt-50 and likely 101. I used them for the SK-ResNeXt-50 32x4d that I trained with 2 GPU using a slightly higher LR per effective batch size (lr=0.18, b=192 per GPU). The cmd line below are tuned for 8 GPU training.


`./distributed_train.sh 8 /imagenet --model resnext50_32x4d --lr 0.6 --warmup-epochs 5 --epochs 240 --weight-decay 1e-4 --sched cosine --reprob 0.4 --recount 3 --remode pixel --aa rand-m7-mstd0.5-inc1 -b 192 -j 6 --amp --dist-bn reduce`

**TODO dig up some more**


## Usage

### Environment

All development and testing has been done in Conda Python 3 environments on Linux x86-64 systems, specifically Python 3.6.x and 3.7.x. Little to no care has been taken to be Python 2.x friendly and I don't plan to support it. If you run into any challenges running on Windows, or other OS, I'm definitely open to looking into those issues so long as it's in a reproducible (read Conda) environment.

PyTorch versions 1.2, 1.3.1, and 1.4 have been tested with this code.

I've tried to keep the dependencies minimal, the setup is as per the PyTorch default install instructions for Conda:
```
conda create -n torch-env
conda activate torch-env
conda install -c pytorch pytorch torchvision cudatoolkit=10.1
conda install pyyaml
```

### Pip
This package can be installed via pip. Currently, the model factory (`timm.create_model`) is the most useful component to use via a pip install.

Install (after conda env/install):
```
pip install timm
```

Use:
```
>>> import timm
>>> m = timm.create_model('mobilenetv3_100', pretrained=True)
>>> m.eval()
```

### Scripts
A train, validation, inference, and checkpoint cleaning script included in the github root folder. Scripts are not currently packaged in the pip release.

#### Training

The variety of training args is large and not all combinations of options (or even options) have been fully tested. For the training dataset folder, specify the folder to the base that contains a `train` and `validation` folder.

To train an SE-ResNet34 on ImageNet, locally distributed, 4 GPUs, one process per GPU w/ cosine schedule, random-erasing prob of 50% and per-pixel random value:

`./distributed_train.sh 4 /data/imagenet --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 -j 4`

NOTE: NVIDIA APEX should be installed to run in per-process distributed via DDP or to enable AMP mixed precision with the --amp flag
 
#### Validation / Inference

Validation and inference scripts are similar in usage. One outputs metrics on a validation set and the other outputs topk class ids in a csv. Specify the folder containing validation images, not the base as in training script. 

To validate with the model's pretrained weights (if they exist):

`python validate.py /imagenet/validation/ --model seresnext26_32x4d --pretrained`

To run inference from a checkpoint:

`python inference.py /imagenet/validation/ --model mobilenetv3_100 --checkpoint ./output/model_best.pth.tar`

## TODO
A number of additions planned in the future for various projects, incl
* Do a model performance (speed + accuracy) benchmarking across all models (make runable as script)
* Complete feature map extraction across all model types and build obj detection/segmentation models and scripts (or integrate backbones with mmdetection, detectron2)
