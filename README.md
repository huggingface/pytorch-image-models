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

Thanks to the following for hardware support:
* TPU Research Cloud (TRC) (https://sites.research.google/trc/about/)
* Nvidia (https://www.nvidia.com/en-us/)

And a big thanks to all GitHub sponsors who helped with some of my costs before I joined Hugging Face.

## What's New

### Oct 10, 2022
* More weights in `maxxvit` series, incl first ConvNeXt block based `coatnext` and `maxxvit` experiments:
  * `coatnext_nano_rw_224` - 82.0 @ 224 (G) -- (uses ConvNeXt conv block, no BatchNorm)
  * `maxxvit_rmlp_nano_rw_256` - 83.0 @ 256, 83.7 @ 320  (G) (uses ConvNeXt conv block, no BN)
  * `maxvit_rmlp_small_rw_224` - 84.5 @ 224, 85.1 @ 320 (G)
  * `maxxvit_rmlp_small_rw_256` - 84.6 @ 256, 84.9 @ 288 (G) -- could be trained better, hparams need tuning (uses ConvNeXt block, no BN)
  * `coatnet_rmlp_2_rw_224` - 84.6 @ 224, 85 @ 320  (T)
  * NOTE: official MaxVit weights (in1k) have been released at https://github.com/google-research/maxvit -- some extra work is needed to port and adapt since my impl was created independently of theirs and has a few small differences + the whole TF same padding fun.
  
### Sept 23, 2022
* LAION-2B CLIP image towers supported as pretrained backbones for fine-tune or features (no classifier)
  * vit_base_patch32_224_clip_laion2b
  * vit_large_patch14_224_clip_laion2b
  * vit_huge_patch14_224_clip_laion2b
  * vit_giant_patch14_224_clip_laion2b

### Sept 7, 2022
* Hugging Face [`timm` docs](https://huggingface.co/docs/hub/timm) home now exists, look for more here in the future
* Add BEiT-v2 weights for base and large 224x224 models from https://github.com/microsoft/unilm/tree/master/beit2
* Add more weights in `maxxvit` series incl a `pico` (7.5M params, 1.9 GMACs), two `tiny` variants:
  * `maxvit_rmlp_pico_rw_256` - 80.5 @ 256, 81.3 @ 320  (T)
  * `maxvit_tiny_rw_224` - 83.5 @ 224 (G)
  * `maxvit_rmlp_tiny_rw_256` - 84.2 @ 256, 84.8 @ 320 (T)

### Aug 29, 2022
* MaxVit window size scales with img_size by default. Add new RelPosMlp MaxViT weight that leverages this:
  * `maxvit_rmlp_nano_rw_256` - 83.0 @ 256, 83.6 @ 320  (T)

### Aug 26, 2022
* CoAtNet (https://arxiv.org/abs/2106.04803) and MaxVit (https://arxiv.org/abs/2204.01697) `timm` original models
  * both found in [`maxxvit.py`](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/maxxvit.py) model def, contains numerous experiments outside scope of original papers
  * an unfinished Tensorflow version from MaxVit authors can be found https://github.com/google-research/maxvit
* Initial CoAtNet and MaxVit timm pretrained weights (working on more):
  * `coatnet_nano_rw_224` - 81.7 @ 224  (T)
  * `coatnet_rmlp_nano_rw_224` - 82.0 @ 224, 82.8 @ 320 (T)
  * `coatnet_0_rw_224` - 82.4  (T)  -- NOTE timm '0' coatnets have 2 more 3rd stage blocks
  * `coatnet_bn_0_rw_224` - 82.4  (T)
  * `maxvit_nano_rw_256` - 82.9 @ 256  (T)
  * `coatnet_rmlp_1_rw_224` - 83.4 @ 224, 84 @ 320  (T)
  * `coatnet_1_rw_224` - 83.6 @ 224 (G) 
  * (T) = TPU trained with `bits_and_tpu` branch training code, (G) = GPU trained
* GCVit (weights adapted from https://github.com/NVlabs/GCVit, code 100% `timm` re-write for license purposes)
* MViT-V2 (multi-scale vit, adapted from https://github.com/facebookresearch/mvit)
* EfficientFormer (adapted from https://github.com/snap-research/EfficientFormer)
* PyramidVisionTransformer-V2 (adapted from https://github.com/whai362/PVT)
* 'Fast Norm' support for LayerNorm and GroupNorm that avoids float32 upcast w/ AMP (uses APEX LN if available for further boost)


### Aug 15, 2022
* ConvNeXt atto weights added
  * `convnext_atto` - 75.7 @ 224, 77.0 @ 288
  * `convnext_atto_ols` - 75.9  @ 224, 77.2 @ 288

### Aug 5, 2022
* More custom ConvNeXt smaller model defs with weights 
  * `convnext_femto` - 77.5 @ 224, 78.7 @ 288
  * `convnext_femto_ols` - 77.9  @ 224, 78.9 @ 288
  * `convnext_pico` - 79.5 @ 224, 80.4 @ 288
  * `convnext_pico_ols` - 79.5 @ 224, 80.5 @ 288
  * `convnext_nano_ols` - 80.9 @ 224, 81.6 @ 288
* Updated EdgeNeXt to improve ONNX export, add new base variant and weights from original (https://github.com/mmaaz60/EdgeNeXt)

### July 28, 2022
* Add freshly minted DeiT-III Medium (width=512, depth=12, num_heads=8) model weights. Thanks [Hugo Touvron](https://github.com/TouvronHugo)!

### July 27, 2022
* All runtime benchmark and validation result csv files are finally up-to-date!
* A few more weights & model defs added:
  * `darknetaa53` -  79.8 @ 256, 80.5 @ 288
  * `convnext_nano` - 80.8 @ 224, 81.5 @ 288
  * `cs3sedarknet_l` - 81.2 @ 256, 81.8 @ 288
  * `cs3darknet_x` - 81.8 @ 256, 82.2 @ 288
  * `cs3sedarknet_x` - 82.2 @ 256, 82.7 @ 288
  * `cs3edgenet_x` - 82.2 @ 256, 82.7 @ 288
  * `cs3se_edgenet_x` - 82.8 @ 256, 83.5 @ 320
* `cs3*` weights above all trained on TPU w/ `bits_and_tpu` branch. Thanks to TRC program! 
* Add output_stride=8 and 16 support to ConvNeXt (dilation)
* deit3 models not being able to resize pos_emb fixed
* Version 0.6.7 PyPi release (/w above bug fixes and new weighs since 0.6.5)

### July 8, 2022
More models, more fixes
* Official research models (w/ weights) added:
  * EdgeNeXt from (https://github.com/mmaaz60/EdgeNeXt)
  * MobileViT-V2 from (https://github.com/apple/ml-cvnets)
  * DeiT III (Revenge of the ViT) from (https://github.com/facebookresearch/deit)
* My own models:
  * Small `ResNet` defs added by request with 1 block repeats for both basic and bottleneck (resnet10 and resnet14)
  * `CspNet` refactored with dataclass config, simplified CrossStage3 (`cs3`) option. These are closer to YOLO-v5+ backbone defs.
  * More relative position vit fiddling. Two `srelpos` (shared relative position) models trained, and a medium w/ class token.
  * Add an alternate downsample mode to EdgeNeXt and train a `small` model. Better than original small, but not their new USI trained weights.
* My own model weight results (all ImageNet-1k training)
  * `resnet10t` - 66.5 @ 176, 68.3 @ 224
  * `resnet14t` - 71.3 @ 176, 72.3 @ 224
  * `resnetaa50` - 80.6 @ 224 , 81.6 @ 288
  * `darknet53` -  80.0 @ 256, 80.5 @ 288
  * `cs3darknet_m` - 77.0 @ 256, 77.6 @ 288
  * `cs3darknet_focus_m` - 76.7 @ 256, 77.3 @ 288
  * `cs3darknet_l` - 80.4 @ 256, 80.9 @ 288
  * `cs3darknet_focus_l` - 80.3 @ 256, 80.9 @ 288
  * `vit_srelpos_small_patch16_224` - 81.1 @ 224, 82.1 @ 320
  * `vit_srelpos_medium_patch16_224` - 82.3 @ 224, 83.1 @ 320
  * `vit_relpos_small_patch16_cls_224` - 82.6 @ 224, 83.6 @ 320
  * `edgnext_small_rw` - 79.6 @ 224, 80.4 @ 320
* `cs3`, `darknet`, and `vit_*relpos` weights above all trained on TPU thanks to TRC program! Rest trained on overheating GPUs.
* Hugging Face Hub support fixes verified, demo notebook TBA
* Pretrained weights / configs can be loaded externally (ie from local disk) w/ support for head adaptation.
* Add support to change image extensions scanned by `timm` datasets/parsers. See (https://github.com/rwightman/pytorch-image-models/pull/1274#issuecomment-1178303103)
* Default ConvNeXt LayerNorm impl to use `F.layer_norm(x.permute(0, 2, 3, 1), ...).permute(0, 3, 1, 2)` via `LayerNorm2d` in all cases. 
  * a bit slower than previous custom impl on some hardware (ie Ampere w/ CL), but overall fewer regressions across wider HW / PyTorch version ranges. 
  * previous impl exists as `LayerNormExp2d` in `models/layers/norm.py`
* Numerous bug fixes
* Currently testing for imminent PyPi 0.6.x release
* LeViT pretraining of larger models still a WIP, they don't train well / easily without distillation. Time to add distill support (finally)?
* ImageNet-22k weight training + finetune ongoing, work on multi-weight support (slowly) chugging along (there are a LOT of weights, sigh) ...

### May 13, 2022
* Official Swin-V2 models and weights added from (https://github.com/microsoft/Swin-Transformer). Cleaned up to support torchscript.
* Some refactoring for existing `timm` Swin-V2-CR impl, will likely do a bit more to bring parts closer to official and decide whether to merge some aspects.
* More Vision Transformer relative position / residual post-norm experiments (all trained on TPU thanks to TRC program)
  * `vit_relpos_small_patch16_224` - 81.5 @ 224, 82.5 @ 320 -- rel pos, layer scale, no class token, avg pool
  * `vit_relpos_medium_patch16_rpn_224` - 82.3 @ 224, 83.1 @ 320 -- rel pos + res-post-norm, no class token, avg pool
  * `vit_relpos_medium_patch16_224` - 82.5 @ 224, 83.3 @ 320 -- rel pos, layer scale, no class token, avg pool
  * `vit_relpos_base_patch16_gapcls_224` - 82.8 @ 224, 83.9 @ 320 -- rel pos, layer scale, class token, avg pool (by mistake)
* Bring 512 dim, 8-head 'medium' ViT model variant back to life (after using in a pre DeiT 'small' model for first ViT impl back in 2020)
* Add ViT relative position support for switching btw existing impl and some additions in official Swin-V2 impl for future trials
* Sequencer2D impl (https://arxiv.org/abs/2205.01972), added via PR from author (https://github.com/okojoalg)

### May 2, 2022
* Vision Transformer experiments adding Relative Position (Swin-V2 log-coord) (`vision_transformer_relpos.py`) and Residual Post-Norm branches (from Swin-V2) (`vision_transformer*.py`)
  * `vit_relpos_base_patch32_plus_rpn_256` - 79.5 @ 256, 80.6 @ 320 -- rel pos + extended width + res-post-norm, no class token, avg pool
  * `vit_relpos_base_patch16_224` - 82.5 @ 224, 83.6 @ 320 -- rel pos, layer scale, no class token, avg pool
  * `vit_base_patch16_rpn_224` - 82.3 @ 224 -- rel pos + res-post-norm, no class token, avg pool
* Vision Transformer refactor to remove representation layer that was only used in initial vit and rarely used since with newer pretrain (ie `How to Train Your ViT`)
* `vit_*` models support removal of class token, use of global average pool, use of fc_norm (ala beit, mae).

### April 22, 2022
* `timm` models are now officially supported in [fast.ai](https://www.fast.ai/)! Just in time for the new Practical Deep Learning course. `timmdocs` documentation link updated to [timm.fast.ai](http://timm.fast.ai/).
* Two more model weights added in the TPU trained [series](https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-tpu-weights). Some In22k pretrain still in progress.
  * `seresnext101d_32x8d` - 83.69 @ 224, 84.35 @ 288
  * `seresnextaa101d_32x8d` (anti-aliased w/ AvgPool2d) - 83.85 @ 224, 84.57 @ 288

### March 23, 2022
* Add `ParallelBlock` and `LayerScale` option to base vit models to support model configs in [Three things everyone should know about ViT](https://arxiv.org/abs/2203.09795)
* `convnext_tiny_hnf` (head norm first) weights trained with (close to) A2 recipe, 82.2% top-1, could do better with more epochs.

### March 21, 2022
* Merge `norm_norm_norm`. **IMPORTANT** this update for a coming 0.6.x release will likely de-stabilize the master branch for a while. Branch [`0.5.x`](https://github.com/rwightman/pytorch-image-models/tree/0.5.x) or a previous 0.5.x release can be used if stability is required.
* Significant weights update (all TPU trained) as described in this [release](https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-tpu-weights)
  * `regnety_040` - 82.3 @ 224, 82.96 @ 288
  * `regnety_064` - 83.0 @ 224, 83.65 @ 288
  * `regnety_080` - 83.17 @ 224, 83.86 @ 288
  * `regnetv_040` - 82.44 @ 224, 83.18 @ 288   (timm pre-act)
  * `regnetv_064` - 83.1 @ 224, 83.71 @ 288   (timm pre-act)
  * `regnetz_040` - 83.67 @ 256, 84.25 @ 320
  * `regnetz_040h` - 83.77 @ 256, 84.5 @ 320 (w/ extra fc in head)
  * `resnetv2_50d_gn` - 80.8 @ 224, 81.96 @ 288 (pre-act GroupNorm)
  * `resnetv2_50d_evos` 80.77 @ 224, 82.04 @ 288 (pre-act EvoNormS)
  * `regnetz_c16_evos`  - 81.9 @ 256, 82.64 @ 320 (EvoNormS)
  * `regnetz_d8_evos`  - 83.42 @ 256, 84.04 @ 320 (EvoNormS)
  * `xception41p` - 82 @ 299   (timm pre-act)
  * `xception65` -  83.17 @ 299
  * `xception65p` -  83.14 @ 299   (timm pre-act)
  * `resnext101_64x4d` - 82.46 @ 224, 83.16 @ 288
  * `seresnext101_32x8d` - 83.57 @ 224, 84.270 @ 288
  * `resnetrs200` - 83.85 @ 256, 84.44 @ 320
* HuggingFace hub support fixed w/ initial groundwork for allowing alternative 'config sources' for pretrained model definitions and weights (generic local file / remote url support soon)
* SwinTransformer-V2 implementation added. Submitted by [Christoph Reich](https://github.com/ChristophReich1996). Training experiments and model changes by myself are ongoing so expect compat breaks.
* Swin-S3 (AutoFormerV2) models / weights added from https://github.com/microsoft/Cream/tree/main/AutoFormerV2
* MobileViT models w/ weights adapted from https://github.com/apple/ml-cvnets
* PoolFormer models w/ weights adapted from https://github.com/sail-sg/poolformer
* VOLO models w/ weights adapted from https://github.com/sail-sg/volo
* Significant work experimenting with non-BatchNorm norm layers such as EvoNorm, FilterResponseNorm, GroupNorm, etc
* Enhance support for alternate norm + act ('NormAct') layers added to a number of models, esp EfficientNet/MobileNetV3, RegNet, and aligned Xception
* Grouped conv support added to EfficientNet family
* Add 'group matching' API to all models to allow grouping model parameters for application of 'layer-wise' LR decay, lr scale added to LR scheduler
* Gradient checkpointing support added to many models
* `forward_head(x, pre_logits=False)` fn added to all models to allow separate calls of `forward_features` + `forward_head`
* All vision transformer and vision MLP models update to return non-pooled / non-token selected features from `foward_features`, for consistency with CNN models, token selection or pooling now applied in `forward_head`

### Feb 2, 2022
* [Chris Hughes](https://github.com/Chris-hughes10) posted an exhaustive run through of `timm` on his blog yesterday. Well worth a read. [Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055)
* I'm currently prepping to merge the `norm_norm_norm` branch back to master (ver 0.6.x) in next week or so.
  * The changes are more extensive than usual and may destabilize and break some model API use (aiming for full backwards compat). So, beware `pip install git+https://github.com/rwightman/pytorch-image-models` installs!
  * `0.5.x` releases and a `0.5.x` branch will remain stable with a cherry pick or two until dust clears. Recommend sticking to pypi install for a bit if you want stable.

### Jan 14, 2022
* Version 0.5.4 w/ release to be pushed to pypi. It's been a while since last pypi update and riskier changes will be merged to main branch soon....
* Add ConvNeXT models /w weights from official impl (https://github.com/facebookresearch/ConvNeXt), a few perf tweaks, compatible with timm features
* Tried training a few small (~1.8-3M param) / mobile optimized models, a few are good so far, more on the way...
  * `mnasnet_small` - 65.6 top-1
  * `mobilenetv2_050` - 65.9
  * `lcnet_100/075/050` - 72.1 / 68.8 / 63.1
  * `semnasnet_075` - 73
  * `fbnetv3_b/d/g` - 79.1 / 79.7 / 82.0
* TinyNet models added by [rsomani95](https://github.com/rsomani95)
* LCNet added via MobileNetV3 architecture

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

## Introduction

Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arxiv papers, etc in the README, documentation, and code docstrings. Please let me know if I missed anything.

## Models

All model architecture families include variants with pretrained weights. There are specific model variants without any weights, it is NOT a bug. Help training new or better weights is always appreciated. Here are some example [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) to get you started.

A full version of the list below with source links can be found in the [documentation](https://rwightman.github.io/pytorch-image-models/models/).

* Aggregating Nested Transformers - https://arxiv.org/abs/2105.12723
* BEiT - https://arxiv.org/abs/2106.08254
* Big Transfer ResNetV2 (BiT) - https://arxiv.org/abs/1912.11370
* Bottleneck Transformers - https://arxiv.org/abs/2101.11605
* CaiT (Class-Attention in Image Transformers) - https://arxiv.org/abs/2103.17239
* CoaT (Co-Scale Conv-Attentional Image Transformers) - https://arxiv.org/abs/2104.06399
* CoAtNet (Convolution and Attention) - https://arxiv.org/abs/2106.04803
* ConvNeXt - https://arxiv.org/abs/2201.03545
* ConViT (Soft Convolutional Inductive Biases Vision Transformers)- https://arxiv.org/abs/2103.10697
* CspNet (Cross-Stage Partial Networks) - https://arxiv.org/abs/1911.11929
* DeiT - https://arxiv.org/abs/2012.12877
* DeiT-III - https://arxiv.org/pdf/2204.07118.pdf
* DenseNet - https://arxiv.org/abs/1608.06993
* DLA - https://arxiv.org/abs/1707.06484
* DPN (Dual-Path Network) - https://arxiv.org/abs/1707.01629
* EdgeNeXt - https://arxiv.org/abs/2206.10589
* EfficientFormer - https://arxiv.org/abs/2206.01191
* EfficientNet (MBConvNet Family)
    * EfficientNet NoisyStudent (B0-B7, L2) - https://arxiv.org/abs/1911.04252
    * EfficientNet AdvProp (B0-B8) - https://arxiv.org/abs/1911.09665
    * EfficientNet (B0-B7) - https://arxiv.org/abs/1905.11946
    * EfficientNet-EdgeTPU (S, M, L) - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
    * EfficientNet V2 - https://arxiv.org/abs/2104.00298
    * FBNet-C - https://arxiv.org/abs/1812.03443
    * MixNet - https://arxiv.org/abs/1907.09595
    * MNASNet B1, A1 (Squeeze-Excite), and Small - https://arxiv.org/abs/1807.11626
    * MobileNet-V2 - https://arxiv.org/abs/1801.04381
    * Single-Path NAS - https://arxiv.org/abs/1904.02877
    * TinyNet - https://arxiv.org/abs/2010.14819
* GCViT (Global Context Vision Transformer) - https://arxiv.org/abs/2206.09959
* GhostNet - https://arxiv.org/abs/1911.11907
* gMLP - https://arxiv.org/abs/2105.08050
* GPU-Efficient Networks - https://arxiv.org/abs/2006.14090
* Halo Nets - https://arxiv.org/abs/2103.12731
* HRNet - https://arxiv.org/abs/1908.07919
* Inception-V3 - https://arxiv.org/abs/1512.00567
* Inception-ResNet-V2 and Inception-V4 - https://arxiv.org/abs/1602.07261
* Lambda Networks - https://arxiv.org/abs/2102.08602
* LeViT (Vision Transformer in ConvNet's Clothing) - https://arxiv.org/abs/2104.01136
* MaxViT (Multi-Axis Vision Transformer) - https://arxiv.org/abs/2204.01697
* MLP-Mixer - https://arxiv.org/abs/2105.01601
* MobileNet-V3 (MBConvNet w/ Efficient Head) - https://arxiv.org/abs/1905.02244
  * FBNet-V3 - https://arxiv.org/abs/2006.02049
  * HardCoRe-NAS - https://arxiv.org/abs/2102.11646
  * LCNet - https://arxiv.org/abs/2109.15099
* MobileViT - https://arxiv.org/abs/2110.02178
* MobileViT-V2 - https://arxiv.org/abs/2206.02680
* MViT-V2 (Improved Multiscale Vision Transformer) - https://arxiv.org/abs/2112.01526
* NASNet-A - https://arxiv.org/abs/1707.07012
* NesT - https://arxiv.org/abs/2105.12723
* NFNet-F - https://arxiv.org/abs/2102.06171
* NF-RegNet / NF-ResNet - https://arxiv.org/abs/2101.08692
* PNasNet - https://arxiv.org/abs/1712.00559
* PoolFormer (MetaFormer) - https://arxiv.org/abs/2111.11418
* Pooling-based Vision Transformer (PiT) - https://arxiv.org/abs/2103.16302
* PVT-V2 (Improved Pyramid Vision Transformer) - https://arxiv.org/abs/2106.13797
* RegNet - https://arxiv.org/abs/2003.13678
* RegNetZ - https://arxiv.org/abs/2103.06877
* RepVGG - https://arxiv.org/abs/2101.03697
* ResMLP - https://arxiv.org/abs/2105.03404
* ResNet/ResNeXt
    * ResNet (v1b/v1.5) - https://arxiv.org/abs/1512.03385
    * ResNeXt - https://arxiv.org/abs/1611.05431
    * 'Bag of Tricks' / Gluon C, D, E, S variations - https://arxiv.org/abs/1812.01187
    * Weakly-supervised (WSL) Instagram pretrained / ImageNet tuned ResNeXt101 - https://arxiv.org/abs/1805.00932
    * Semi-supervised (SSL) / Semi-weakly Supervised (SWSL) ResNet/ResNeXts - https://arxiv.org/abs/1905.00546
    * ECA-Net (ECAResNet) - https://arxiv.org/abs/1910.03151v4
    * Squeeze-and-Excitation Networks (SEResNet) - https://arxiv.org/abs/1709.01507
    * ResNet-RS - https://arxiv.org/abs/2103.07579
* Res2Net - https://arxiv.org/abs/1904.01169
* ResNeSt - https://arxiv.org/abs/2004.08955
* ReXNet - https://arxiv.org/abs/2007.00992
* SelecSLS - https://arxiv.org/abs/1907.00837
* Selective Kernel Networks - https://arxiv.org/abs/1903.06586
* Sequencer2D - https://arxiv.org/abs/2205.01972
* Swin S3 (AutoFormerV2) - https://arxiv.org/abs/2111.14725
* Swin Transformer - https://arxiv.org/abs/2103.14030
* Swin Transformer V2 - https://arxiv.org/abs/2111.09883
* Transformer-iN-Transformer (TNT) - https://arxiv.org/abs/2103.00112
* TResNet - https://arxiv.org/abs/2003.13630
* Twins (Spatial Attention in Vision Transformers) - https://arxiv.org/pdf/2104.13840.pdf
* Visformer - https://arxiv.org/abs/2104.12533
* Vision Transformer - https://arxiv.org/abs/2010.11929
* VOLO (Vision Outlooker) - https://arxiv.org/abs/2106.13112
* VovNet V2 and V1 - https://arxiv.org/abs/1911.06667
* Xception - https://arxiv.org/abs/1610.02357
* Xception (Modified Aligned, Gluon) - https://arxiv.org/abs/1802.02611
* Xception (Modified Aligned, TF) - https://arxiv.org/abs/1802.02611
* XCiT (Cross-Covariance Image Transformers) - https://arxiv.org/abs/2106.09681

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
* A 'Test Time Pool' wrapper that can wrap any of the included models and usually provides improved performance doing inference with input images larger than the training size. Idea adapted from original DPN implementation when I ported (https://github.com/cypw/DPNs)
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
* Blur Pooling (https://arxiv.org/abs/1904.11486)
* Space-to-Depth by [mrT23](https://github.com/mrT23/TResNet/blob/master/src/models/tresnet/layers/space_to_depth.py) (https://arxiv.org/abs/1801.04590) -- original paper?
* Adaptive Gradient Clipping (https://arxiv.org/abs/2102.06171, https://github.com/deepmind/deepmind-research/tree/master/nfnets)
* An extensive selection of channel and/or spatial attention modules:
    * Bottleneck Transformer - https://arxiv.org/abs/2101.11605
    * CBAM - https://arxiv.org/abs/1807.06521
    * Effective Squeeze-Excitation (ESE) - https://arxiv.org/abs/1911.06667
    * Efficient Channel Attention (ECA) - https://arxiv.org/abs/1910.03151
    * Gather-Excite (GE) - https://arxiv.org/abs/1810.12348
    * Global Context (GC) - https://arxiv.org/abs/1904.11492
    * Halo - https://arxiv.org/abs/2103.12731
    * Involution - https://arxiv.org/abs/2103.06255
    * Lambda Layer - https://arxiv.org/abs/2102.08602
    * Non-Local (NL) -  https://arxiv.org/abs/1711.07971
    * Squeeze-and-Excitation (SE) - https://arxiv.org/abs/1709.01507
    * Selective Kernel (SK) - (https://arxiv.org/abs/1903.06586
    * Split (SPLAT) - https://arxiv.org/abs/2004.08955
    * Shifted Window (SWIN) - https://arxiv.org/abs/2103.14030

## Results

Model validation results can be found in the [documentation](https://rwightman.github.io/pytorch-image-models/results/) and in the [results tables](results/README.md)

## Getting Started (Documentation)

My current [documentation](https://rwightman.github.io/pytorch-image-models/) for `timm` covers the basics.

Hugging Face [`timm` docs](https://huggingface.co/docs/hub/timm) will be the documentation focus going forward and will eventually replace the `github.io` docs above.

[Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055) by [Chris Hughes](https://github.com/Chris-hughes10) is an extensive blog post covering many aspects of `timm` in detail.

[timmdocs](http://timm.fast.ai/) is quickly becoming a much more comprehensive set of documentation for `timm`. A big thanks to [Aman Arora](https://github.com/amaarora) for his efforts creating timmdocs.

[paperswithcode](https://paperswithcode.com/lib/timm) is a good resource for browsing the models within `timm`.

## Train, Validation, Inference Scripts

The root folder of the repository contains reference train, validation, and inference scripts that work with the included models and other features of this repository. They are adaptable for other datasets and use cases with a little hacking. See [documentation](https://rwightman.github.io/pytorch-image-models/scripts/) for some basics and [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) for some train examples that produce SOTA ImageNet results.

## Awesome PyTorch Resources

One of the greatest assets of PyTorch is the community and their contributions. A few of my favourite resources that pair well with the models and components here are listed below.

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
The code here is licensed Apache 2.0. I've taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc. I've made an effort to avoid any GPL / LGPL conflicts. That said, it is your responsibility to ensure you comply with licenses here and conditions of any dependent licenses. Where applicable, I've linked the sources/references for various components in docstrings. If you think I've missed anything please create an issue.

### Pretrained Weights
So far all of the pretrained weights available here are pretrained on ImageNet with a select few that have some additional pretraining (see extra note below). ImageNet was released for non-commercial research purposes only (https://image-net.org/download). It's not clear what the implications of that are for the use of pretrained weights from that dataset. Any models I have trained with ImageNet are done for research purposes and one should assume that the original dataset license applies to the weights. It's best to seek legal advice if you intend to use the pretrained weights in a commercial product.

#### Pretrained on more than ImageNet
Several weights included or references here were pretrained with proprietary datasets that I do not have access to. These include the Facebook WSL, SSL, SWSL ResNe(Xt) and the Google Noisy Student EfficientNet models. The Facebook models have an explicit non-commercial license (CC-BY-NC 4.0, https://github.com/facebookresearch/semi-supervised-ImageNet1K-models, https://github.com/facebookresearch/WSL-Images). The Google models do not appear to have any restriction beyond the Apache 2.0 license (and ImageNet concerns). In either case, you should contact Facebook or Google with any questions.

## Citing

### BibTeX

```bibtex
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
