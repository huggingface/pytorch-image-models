# Archived Changes

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

### Feb 6, 2020
* Add RandAugment trained EfficientNet-ES (EdgeTPU-Small) weights with 78.1 top-1. Trained by [Andrew Lavin](https://github.com/andravin) (see Training section for hparams)

### Feb 1/2, 2020
* Port new EfficientNet-B8 (RandAugment) weights, these are different than the B8 AdvProp, different input normalization.
* Update results csv files on all models for ImageNet validation and three other test sets
* Push PyPi package update

### Jan 31, 2020
* Update ResNet50 weights with a new 79.038 result from further JSD / AugMix experiments. Full command line for reproduction in training section below.

### Jan 11/12, 2020
* Master may be a bit unstable wrt to training, these changes have been tested but not all combos
* Implementations of AugMix added to existing RA and AA. Including numerous supporting pieces like JSD loss (Jensen-Shannon divergence + CE), and AugMixDataset
* SplitBatchNorm adaptation layer added for implementing Auxiliary BN as per AdvProp paper
* ResNet-50 AugMix trained model w/ 79% top-1 added
* `seresnext26tn_32x4d` - 77.99 top-1, 93.75 top-5 added to tiered experiment, higher img/s than 't' and 'd'

### Jan 3, 2020
* Add RandAugment trained EfficientNet-B0 weight with 77.7 top-1. Trained by [Michael Klachko](https://github.com/michaelklachko) with this code and recent hparams (see Training section)
* Add `avg_checkpoints.py` script for post training weight averaging and update all scripts with header docstrings and shebangs.

### Dec 30, 2019
* Merge [Dushyant Mehta's](https://github.com/mehtadushy) PR for SelecSLS (Selective Short and Long Range Skip Connections) networks. Good GPU memory consumption and throughput. Original: https://github.com/mehtadushy/SelecSLS-Pytorch

### Dec 28, 2019
* Add new model weights and training hparams (see Training Hparams section)
  * `efficientnet_b3` - 81.5 top-1, 95.7 top-5 at default res/crop, 81.9, 95.8 at 320x320 1.0 crop-pct
     * trained with RandAugment, ended up with an interesting but less than perfect result (see training section)
  * `seresnext26d_32x4d`- 77.6 top-1, 93.6 top-5
     * deep stem (32, 32, 64), avgpool downsample
     * stem/dowsample from bag-of-tricks paper
  * `seresnext26t_32x4d`- 78.0 top-1, 93.7 top-5
     * deep tiered stem (24, 48, 64), avgpool downsample (a modified 'D' variant)
     * stem sizing mods from Jeremy Howard and fastai devs discussing ResNet architecture experiments

### Dec 23, 2019
* Add RandAugment trained MixNet-XL weights with 80.48 top-1.
* `--dist-bn` argument added to train.py, will distribute BN stats between nodes after each train epoch, before eval

### Dec 4, 2019
* Added weights from the first training from scratch of an EfficientNet (B2) with my new RandAugment implementation. Much better than my previous B2 and very close to the official AdvProp ones (80.4 top-1, 95.08 top-5).

### Nov 29, 2019
* Brought EfficientNet and MobileNetV3 up to date with my https://github.com/rwightman/gen-efficientnet-pytorch code. Torchscript and ONNX export compat excluded.
  * AdvProp weights added
  * Official TF MobileNetv3 weights added
* EfficientNet and MobileNetV3 hook based 'feature extraction' classes added. Will serve as basis for using models as backbones in obj detection/segmentation tasks. Lots more to be done here...
* HRNet classification models and weights added from https://github.com/HRNet/HRNet-Image-Classification
* Consistency in global pooling, `reset_classifer`, and `forward_features` across models
  * `forward_features` always returns unpooled feature maps now
* Reasonable chance I broke something... let me know

### Nov 22, 2019
* Add ImageNet training RandAugment implementation alongside AutoAugment. PyTorch Transform compatible format, using PIL. Currently training two EfficientNet models from scratch with promising results... will update.
* `drop-connect` cmd line arg finally added to `train.py`, no need to hack model fns. Works for efficientnet/mobilenetv3 based models, ignored otherwise.