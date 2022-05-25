# Recent Changes


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
