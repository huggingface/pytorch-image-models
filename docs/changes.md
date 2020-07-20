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
