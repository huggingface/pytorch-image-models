# Archived Changes

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