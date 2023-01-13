# Archived Changes

### Nov 22, 2021
* A number of updated weights anew new model defs
  * `eca_halonext26ts` - 79.5 @ 256
  * `resnet50_gn` (new) - 80.1 @ 224, 81.3 @ 288
  * `resnet50` - 80.7 @ 224, 80.9 @ 288 (trained at 176, not replacing current a1 weights as default since these don't scale as well to higher res, [weights](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth))
  * `resnext50_32x4d` - 81.1 @ 224, 82.0 @ 288
  * `sebotnet33ts_256` (new) - 81.2 @ 224
  * `lamhalobotnet50ts_256` - 81.5 @ 256
  * `halonet50ts` - 81.7 @ 256
  * `halo2botnet50ts_256` - 82.0 @ 256
  * `resnet101` - 82.0 @ 224, 82.8 @ 288
  * `resnetv2_101` (new) - 82.1 @ 224, 83.0 @ 288
  * `resnet152` - 82.8 @ 224, 83.5 @ 288
  * `regnetz_d8` (new) - 83.5 @ 256, 84.0 @ 320
  * `regnetz_e8` (new) - 84.5 @ 256, 85.0 @ 320
* `vit_base_patch8_224` (85.8 top-1) & `in21k` variant weights added thanks [Martins Bruveris](https://github.com/martinsbruveris)
* Groundwork in for FX feature extraction thanks to [Alexander Soare](https://github.com/alexander-soare)
  * models updated for tracing compatibility (almost full support with some distlled transformer exceptions)

### Oct 19, 2021
* ResNet strikes back (https://arxiv.org/abs/2110.00476) weights added, plus any extra training components used. Model weights and some more details here (https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-rsb-weights)
* BCE loss and Repeated Augmentation support for RSB paper
* 4 series of ResNet based attention model experiments being added (implemented across byobnet.py/byoanet.py). These include all sorts of attention, from channel attn like SE, ECA to 2D QKV self-attention layers such as Halo, Bottlneck, Lambda. Details here (https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-attn-weights)
* Working implementations of the following 2D self-attention modules (likely to be differences from paper or eventual official impl):
  * Halo (https://arxiv.org/abs/2103.12731)
  * Bottleneck Transformer (https://arxiv.org/abs/2101.11605)
  * LambdaNetworks (https://arxiv.org/abs/2102.08602)
* A RegNetZ series of models with some attention experiments (being added to). These do not follow the paper (https://arxiv.org/abs/2103.06877) in any way other than block architecture, details of official models are not available. See more here (https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-attn-weights)
* ConvMixer (https://openreview.net/forum?id=TVHS5Y4dNvM), CrossVit (https://arxiv.org/abs/2103.14899), and BeiT (https://arxiv.org/abs/2106.08254) architectures + weights added
* freeze/unfreeze helpers by [Alexander Soare](https://github.com/alexander-soare)

### Aug 18, 2021
* Optimizer bonanza!
  * Add LAMB and LARS optimizers, incl trust ratio clipping options. Tweaked to work properly in PyTorch XLA (tested on TPUs w/ `timm bits` [branch](https://github.com/rwightman/pytorch-image-models/tree/bits_and_tpu/timm/bits))
  * Add MADGRAD from FB research w/ a few tweaks (decoupled decay option, step handling that works with PyTorch XLA)
  * Some cleanup on all optimizers and factory. No more `.data`, a bit more consistency, unit tests for all!
  * SGDP and AdamP still won't work with PyTorch XLA but others should (have yet to test Adabelief, Adafactor, Adahessian myself).
* EfficientNet-V2 XL TF ported weights added, but they don't validate well in PyTorch (L is better). The pre-processing for the V2 TF training is a bit diff and the fine-tuned 21k -> 1k weights are very sensitive and less robust than the 1k weights.
* Added PyTorch trained EfficientNet-V2 'Tiny' w/ GlobalContext attn weights. Only .1-.2 top-1 better than the SE so more of a curiosity for those interested.

### July 12, 2021
* Add XCiT models from [official facebook impl](https://github.com/facebookresearch/xcit). Contributed by [Alexander Soare](https://github.com/alexander-soare)

### July 5-9, 2021
* Add `efficientnetv2_rw_t` weights, a custom 'tiny' 13.6M param variant that is a bit better than (non NoisyStudent) B3 models. Both faster and better accuracy (at same or lower res)
  * top-1 82.34 @ 288x288 and 82.54 @ 320x320
* Add [SAM pretrained](https://arxiv.org/abs/2106.01548) in1k weight for ViT B/16 (`vit_base_patch16_sam_224`) and B/32 (`vit_base_patch32_sam_224`)  models.
* Add 'Aggregating Nested Transformer' (NesT) w/ weights converted from official [Flax impl](https://github.com/google-research/nested-transformer). Contributed by [Alexander Soare](https://github.com/alexander-soare).
  * `jx_nest_base` - 83.534, `jx_nest_small` - 83.120, `jx_nest_tiny` - 81.426

### June 23, 2021
* Reproduce gMLP model training, `gmlp_s16_224` trained to 79.6 top-1, matching [paper](https://arxiv.org/abs/2105.08050). Hparams for this and other recent MLP training [here](https://gist.github.com/rwightman/d6c264a9001f9167e06c209f630b2cc6)

### June 20, 2021
* Release Vision Transformer 'AugReg' weights from [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
  * .npz weight loading support added, can load any of the 50K+ weights from the [AugReg series](https://console.cloud.google.com/storage/browser/vit_models/augreg)
  * See [example notebook](https://colab.research.google.com/github/google-research/vision_transformer/blob/master/vit_jax_augreg.ipynb) from [official impl](https://github.com/google-research/vision_transformer/) for navigating the augreg weights
  * Replaced all default weights w/ best AugReg variant (if possible). All AugReg 21k classifiers work.
    * Highlights: `vit_large_patch16_384` (87.1 top-1), `vit_large_r50_s32_384` (86.2 top-1), `vit_base_patch16_384` (86.0 top-1)
  * `vit_deit_*` renamed to just `deit_*`
  * Remove my old small model, replace with DeiT compatible small w/ AugReg weights
* Add 1st training of my `gmixer_24_224` MLP /w GLU, 78.1 top-1 w/ 25M params.
* Add weights from official ResMLP release (https://github.com/facebookresearch/deit)
* Add `eca_nfnet_l2` weights from my 'lightweight' series. 84.7 top-1 at 384x384.
* Add distilled BiT 50x1 student and 152x2 Teacher weights from  [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237)
* NFNets and ResNetV2-BiT models work w/ Pytorch XLA now
  * weight standardization uses F.batch_norm instead of std_mean (std_mean wasn't lowered)
  * eps values adjusted, will be slight differences but should be quite close
* Improve test coverage and classifier interface of non-conv (vision transformer and mlp) models
* Cleanup a few classifier / flatten details for models w/ conv classifiers or early global pool
* Please report any regressions, this PR touched quite a few models.

### June 8, 2021
* Add first ResMLP weights, trained in PyTorch XLA on TPU-VM w/ my XLA branch. 24 block variant, 79.2 top-1.
* Add ResNet51-Q model w/ pretrained weights at 82.36 top-1.
  * NFNet inspired block layout with quad layer stem and no maxpool
  * Same param count (35.7M) and throughput as ResNetRS-50 but +1.5 top-1 @ 224x224 and +2.5 top-1 at 288x288

### May 25, 2021
* Add LeViT, Visformer, Convit (PR by Aman Arora), Twins (PR by paper authors) transformer models
* Cleanup input_size/img_size override handling and testing for all vision transformer models
* Add `efficientnetv2_rw_m` model and weights (started training before official code). 84.8 top-1, 53M params.

### May 14, 2021
* Add EfficientNet-V2 official model defs w/ ported weights from official [Tensorflow/Keras](https://github.com/google/automl/tree/master/efficientnetv2) impl.
  * 1k trained variants: `tf_efficientnetv2_s/m/l`
  * 21k trained variants: `tf_efficientnetv2_s/m/l_in21k`
  * 21k pretrained -> 1k fine-tuned: `tf_efficientnetv2_s/m/l_in21ft1k`
  * v2 models w/ v1 scaling: `tf_efficientnetv2_b0` through `b3`
  * Rename my prev V2 guess `efficientnet_v2s` -> `efficientnetv2_rw_s`
  * Some blank `efficientnetv2_*` models in-place for future native PyTorch training

### May 5, 2021
* Add MLP-Mixer models and port pretrained weights from [Google JAX impl](https://github.com/google-research/vision_transformer/tree/linen)
* Add CaiT models and pretrained weights from [FB](https://github.com/facebookresearch/deit)
* Add ResNet-RS models and weights from [TF](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs). Thanks [Aman Arora](https://github.com/amaarora)
* Add CoaT models and weights. Thanks [Mohammed Rizin](https://github.com/morizin)
* Add new ImageNet-21k weights & finetuned weights for TResNet, MobileNet-V3, ViT models. Thanks [mrT](https://github.com/mrT23)
* Add GhostNet models and weights. Thanks [Kai Han](https://github.com/iamhankai)
* Update ByoaNet attention modles
   * Improve SA module inits
   * Hack together experimental stand-alone Swin based attn module and `swinnet`
   * Consistent '26t' model defs for experiments.
* Add improved Efficientnet-V2S (prelim model def) weights. 83.8 top-1.
* WandB logging support

### April 13, 2021
* Add Swin Transformer models and weights from https://github.com/microsoft/Swin-Transformer

### April 12, 2021
* Add ECA-NFNet-L1 (slimmed down F1 w/ SiLU, 41M params) trained with this code. 84% top-1 @ 320x320. Trained at 256x256.
* Add EfficientNet-V2S model (unverified model definition) weights. 83.3 top-1 @ 288x288. Only trained single res 224. Working on progressive training.
* Add ByoaNet model definition (Bring-your-own-attention) w/ SelfAttention block and corresponding SA/SA-like modules and model defs
  * Lambda Networks - https://arxiv.org/abs/2102.08602
  * Bottleneck Transformers - https://arxiv.org/abs/2101.11605
  * Halo Nets - https://arxiv.org/abs/2103.12731
* Adabelief optimizer contributed by Juntang Zhuang

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

### March 7, 2021
* First 0.4.x PyPi release w/ NFNets (& related), ByoB (GPU-Efficient, RepVGG, etc).
* Change feature extraction for pre-activation nets (NFNets, ResNetV2) to return features before activation.

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
* More model archs, incl a flexible ByobNet backbone ('Bring-your-own-blocks')
  * GPU-Efficient-Networks (https://github.com/idstcv/GPU-Efficient-Networks), impl in `byobnet.py`
  * RepVGG (https://github.com/DingXiaoH/RepVGG), impl in `byobnet.py`
  * classic VGG (from torchvision, impl in `vgg`)
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
* Update results files

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