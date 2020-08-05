# Recent Changes

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
  * EfficientNet-Lite0 (non-TF ver) - 75.5 (submitted by @hal-314)
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
