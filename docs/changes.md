# Recent Changes

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

* All models support the `features_only=True` argument for `create_model` call to return a network that extracts features from the deepest layer at each stride.
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
