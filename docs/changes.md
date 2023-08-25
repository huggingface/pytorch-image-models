# Recent Changes

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
* Add support to change image extensions scanned by `timm` datasets/readers. See (https://github.com/rwightman/pytorch-image-models/pull/1274#issuecomment-1178303103)
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

### Jan 5, 2023
* ConvNeXt-V2 models and weights added to existing `convnext.py`
  * Paper: [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](http://arxiv.org/abs/2301.00808)
  * Reference impl: https://github.com/facebookresearch/ConvNeXt-V2 (NOTE: weights currently CC-BY-NC)

### Dec 23, 2022 🎄☃
* Add FlexiViT models and weights from https://github.com/google-research/big_vision (check out paper at https://arxiv.org/abs/2212.08013)
  * NOTE currently resizing is static on model creation, on-the-fly dynamic / train patch size sampling is a WIP
* Many more models updated to multi-weight and downloadable via HF hub now (convnext, efficientnet, mobilenet, vision_transformer*, beit)
* More model pretrained tag and adjustments, some model names changed (working on deprecation translations, consider main branch DEV branch right now, use 0.6.x for stable use)
* More ImageNet-12k (subset of 22k) pretrain models popping up:
  * `efficientnet_b5.in12k_ft_in1k` - 85.9 @ 448x448
  * `vit_medium_patch16_gap_384.in12k_ft_in1k` - 85.5 @ 384x384
  * `vit_medium_patch16_gap_256.in12k_ft_in1k` - 84.5 @ 256x256
  * `convnext_nano.in12k_ft_in1k` - 82.9 @ 288x288

### Dec 8, 2022
* Add 'EVA l' to `vision_transformer.py`, MAE style ViT-L/14 MIM pretrain w/ EVA-CLIP targets, FT on ImageNet-1k (w/ ImageNet-22k intermediate for some)
  * original source: https://github.com/baaivision/EVA

| model                                     | top1 | param_count |  gmac | macts | hub                                     |
|:------------------------------------------|-----:|------------:|------:|------:|:----------------------------------------|
| eva_large_patch14_336.in22k_ft_in22k_in1k | 89.2 |       304.5 | 191.1 | 270.2 | [link](https://huggingface.co/BAAI/EVA) |
| eva_large_patch14_336.in22k_ft_in1k       | 88.7 |       304.5 | 191.1 | 270.2 | [link](https://huggingface.co/BAAI/EVA) |
| eva_large_patch14_196.in22k_ft_in22k_in1k | 88.6 |       304.1 |  61.6 |  63.5 | [link](https://huggingface.co/BAAI/EVA) |
| eva_large_patch14_196.in22k_ft_in1k       | 87.9 |       304.1 |  61.6 |  63.5 | [link](https://huggingface.co/BAAI/EVA) |

### Dec 6, 2022
* Add 'EVA g', BEiT style ViT-g/14 model weights w/ both MIM pretrain and CLIP pretrain to `beit.py`. 
  * original source: https://github.com/baaivision/EVA
  * paper: https://arxiv.org/abs/2211.07636

| model                                    |   top1 |   param_count |   gmac |   macts | hub                                     |
|:-----------------------------------------|-------:|--------------:|-------:|--------:|:----------------------------------------|
| eva_giant_patch14_560.m30m_ft_in22k_in1k |   89.8 |        1014.4 | 1906.8 |  2577.2 | [link](https://huggingface.co/BAAI/EVA) |
| eva_giant_patch14_336.m30m_ft_in22k_in1k |   89.6 |        1013   |  620.6 |   550.7 | [link](https://huggingface.co/BAAI/EVA) |
| eva_giant_patch14_336.clip_ft_in1k       |   89.4 |        1013   |  620.6 |   550.7 | [link](https://huggingface.co/BAAI/EVA) |
| eva_giant_patch14_224.clip_ft_in1k       |   89.1 |        1012.6 |  267.2 |   192.6 | [link](https://huggingface.co/BAAI/EVA) |

### Dec 5, 2022

* Pre-release (`0.8.0dev0`) of multi-weight support (`model_arch.pretrained_tag`). Install with `pip install --pre timm`
  * vision_transformer, maxvit, convnext are the first three model impl w/ support
  * model names are changing with this (previous _21k, etc. fn will merge), still sorting out deprecation handling
  * bugs are likely, but I need feedback so please try it out
  * if stability is needed, please use 0.6.x pypi releases or clone from [0.6.x branch](https://github.com/rwightman/pytorch-image-models/tree/0.6.x)
* Support for PyTorch 2.0 compile is added in train/validate/inference/benchmark, use `--torchcompile` argument
* Inference script allows more control over output, select k for top-class index + prob json, csv or parquet output
* Add a full set of fine-tuned CLIP image tower weights from both LAION-2B and original OpenAI CLIP models

| model                                            |   top1 |   param_count |   gmac |   macts | hub                                                                                  |
|:-------------------------------------------------|-------:|--------------:|-------:|--------:|:-------------------------------------------------------------------------------------|
| vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k  |   88.6 |         632.5 |  391   |   407.5 | [link](https://huggingface.co/timm/vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k)  |
| vit_large_patch14_clip_336.openai_ft_in12k_in1k  |   88.3 |         304.5 |  191.1 |   270.2 | [link](https://huggingface.co/timm/vit_large_patch14_clip_336.openai_ft_in12k_in1k)  |
| vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k  |   88.2 |         632   |  167.4 |   139.4 | [link](https://huggingface.co/timm/vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k)  |
| vit_large_patch14_clip_336.laion2b_ft_in12k_in1k |   88.2 |         304.5 |  191.1 |   270.2 | [link](https://huggingface.co/timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k) |
| vit_large_patch14_clip_224.openai_ft_in12k_in1k  |   88.2 |         304.2 |   81.1 |    88.8 | [link](https://huggingface.co/timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k)  |
| vit_large_patch14_clip_224.laion2b_ft_in12k_in1k |   87.9 |         304.2 |   81.1 |    88.8 | [link](https://huggingface.co/timm/vit_large_patch14_clip_224.laion2b_ft_in12k_in1k) |
| vit_large_patch14_clip_224.openai_ft_in1k        |   87.9 |         304.2 |   81.1 |    88.8 | [link](https://huggingface.co/timm/vit_large_patch14_clip_224.openai_ft_in1k)        |
| vit_large_patch14_clip_336.laion2b_ft_in1k       |   87.9 |         304.5 |  191.1 |   270.2 | [link](https://huggingface.co/timm/vit_large_patch14_clip_336.laion2b_ft_in1k)       |
| vit_huge_patch14_clip_224.laion2b_ft_in1k        |   87.6 |         632   |  167.4 |   139.4 | [link](https://huggingface.co/timm/vit_huge_patch14_clip_224.laion2b_ft_in1k)        |
| vit_large_patch14_clip_224.laion2b_ft_in1k       |   87.3 |         304.2 |   81.1 |    88.8 | [link](https://huggingface.co/timm/vit_large_patch14_clip_224.laion2b_ft_in1k)       |
| vit_base_patch16_clip_384.laion2b_ft_in12k_in1k  |   87.2 |          86.9 |   55.5 |   101.6 | [link](https://huggingface.co/timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k)  |
| vit_base_patch16_clip_384.openai_ft_in12k_in1k   |   87   |          86.9 |   55.5 |   101.6 | [link](https://huggingface.co/timm/vit_base_patch16_clip_384.openai_ft_in12k_in1k)   |
| vit_base_patch16_clip_384.laion2b_ft_in1k        |   86.6 |          86.9 |   55.5 |   101.6 | [link](https://huggingface.co/timm/vit_base_patch16_clip_384.laion2b_ft_in1k)        |
| vit_base_patch16_clip_384.openai_ft_in1k         |   86.2 |          86.9 |   55.5 |   101.6 | [link](https://huggingface.co/timm/vit_base_patch16_clip_384.openai_ft_in1k)         |
| vit_base_patch16_clip_224.laion2b_ft_in12k_in1k  |   86.2 |          86.6 |   17.6 |    23.9 | [link](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k)  |
| vit_base_patch16_clip_224.openai_ft_in12k_in1k   |   85.9 |          86.6 |   17.6 |    23.9 | [link](https://huggingface.co/timm/vit_base_patch16_clip_224.openai_ft_in12k_in1k)   |
| vit_base_patch32_clip_448.laion2b_ft_in12k_in1k  |   85.8 |          88.3 |   17.9 |    23.9 | [link](https://huggingface.co/timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k)  |
| vit_base_patch16_clip_224.laion2b_ft_in1k        |   85.5 |          86.6 |   17.6 |    23.9 | [link](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b_ft_in1k)        |
| vit_base_patch32_clip_384.laion2b_ft_in12k_in1k  |   85.4 |          88.3 |   13.1 |    16.5 | [link](https://huggingface.co/timm/vit_base_patch32_clip_384.laion2b_ft_in12k_in1k)  |
| vit_base_patch16_clip_224.openai_ft_in1k         |   85.3 |          86.6 |   17.6 |    23.9 | [link](https://huggingface.co/timm/vit_base_patch16_clip_224.openai_ft_in1k)         |
| vit_base_patch32_clip_384.openai_ft_in12k_in1k   |   85.2 |          88.3 |   13.1 |    16.5 | [link](https://huggingface.co/timm/vit_base_patch32_clip_384.openai_ft_in12k_in1k)   |
| vit_base_patch32_clip_224.laion2b_ft_in12k_in1k  |   83.3 |          88.2 |    4.4 |     5   | [link](https://huggingface.co/timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k)  |
| vit_base_patch32_clip_224.laion2b_ft_in1k        |   82.6 |          88.2 |    4.4 |     5   | [link](https://huggingface.co/timm/vit_base_patch32_clip_224.laion2b_ft_in1k)        |
| vit_base_patch32_clip_224.openai_ft_in1k         |   81.9 |          88.2 |    4.4 |     5   | [link](https://huggingface.co/timm/vit_base_patch32_clip_224.openai_ft_in1k)         |

* Port of MaxViT Tensorflow Weights from official impl at https://github.com/google-research/maxvit
  * There was larger than expected drops for the upscaled 384/512 in21k fine-tune weights, possible detail missing, but the 21k FT did seem sensitive to small preprocessing

| model                              |   top1 |   param_count |   gmac |   macts | hub                                                                    |
|:-----------------------------------|-------:|--------------:|-------:|--------:|:-----------------------------------------------------------------------|
| maxvit_xlarge_tf_512.in21k_ft_in1k |   88.5 |         475.8 |  534.1 |  1413.2 | [link](https://huggingface.co/timm/maxvit_xlarge_tf_512.in21k_ft_in1k) |
| maxvit_xlarge_tf_384.in21k_ft_in1k |   88.3 |         475.3 |  292.8 |   668.8 | [link](https://huggingface.co/timm/maxvit_xlarge_tf_384.in21k_ft_in1k) |
| maxvit_base_tf_512.in21k_ft_in1k   |   88.2 |         119.9 |  138   |   704   | [link](https://huggingface.co/timm/maxvit_base_tf_512.in21k_ft_in1k)   |
| maxvit_large_tf_512.in21k_ft_in1k  |   88   |         212.3 |  244.8 |   942.2 | [link](https://huggingface.co/timm/maxvit_large_tf_512.in21k_ft_in1k)  |
| maxvit_large_tf_384.in21k_ft_in1k  |   88   |         212   |  132.6 |   445.8 | [link](https://huggingface.co/timm/maxvit_large_tf_384.in21k_ft_in1k)  |
| maxvit_base_tf_384.in21k_ft_in1k   |   87.9 |         119.6 |   73.8 |   332.9 | [link](https://huggingface.co/timm/maxvit_base_tf_384.in21k_ft_in1k)   |
| maxvit_base_tf_512.in1k            |   86.6 |         119.9 |  138   |   704   | [link](https://huggingface.co/timm/maxvit_base_tf_512.in1k)            |
| maxvit_large_tf_512.in1k           |   86.5 |         212.3 |  244.8 |   942.2 | [link](https://huggingface.co/timm/maxvit_large_tf_512.in1k)           |
| maxvit_base_tf_384.in1k            |   86.3 |         119.6 |   73.8 |   332.9 | [link](https://huggingface.co/timm/maxvit_base_tf_384.in1k)            |
| maxvit_large_tf_384.in1k           |   86.2 |         212   |  132.6 |   445.8 | [link](https://huggingface.co/timm/maxvit_large_tf_384.in1k)           |
| maxvit_small_tf_512.in1k           |   86.1 |          69.1 |   67.3 |   383.8 | [link](https://huggingface.co/timm/maxvit_small_tf_512.in1k)           |
| maxvit_tiny_tf_512.in1k            |   85.7 |          31   |   33.5 |   257.6 | [link](https://huggingface.co/timm/maxvit_tiny_tf_512.in1k)            |
| maxvit_small_tf_384.in1k           |   85.5 |          69   |   35.9 |   183.6 | [link](https://huggingface.co/timm/maxvit_small_tf_384.in1k)           |
| maxvit_tiny_tf_384.in1k            |   85.1 |          31   |   17.5 |   123.4 | [link](https://huggingface.co/timm/maxvit_tiny_tf_384.in1k)            |
| maxvit_large_tf_224.in1k           |   84.9 |         211.8 |   43.7 |   127.4 | [link](https://huggingface.co/timm/maxvit_large_tf_224.in1k)           |
| maxvit_base_tf_224.in1k            |   84.9 |         119.5 |   24   |    95   | [link](https://huggingface.co/timm/maxvit_base_tf_224.in1k)            |
| maxvit_small_tf_224.in1k           |   84.4 |          68.9 |   11.7 |    53.2 | [link](https://huggingface.co/timm/maxvit_small_tf_224.in1k)           |
| maxvit_tiny_tf_224.in1k            |   83.4 |          30.9 |    5.6 |    35.8 | [link](https://huggingface.co/timm/maxvit_tiny_tf_224.in1k)            |

### Oct 15, 2022
* Train and validation script enhancements
* Non-GPU (ie CPU) device support
* SLURM compatibility for train script
* HF datasets support (via ReaderHfds)
* TFDS/WDS dataloading improvements (sample padding/wrap for distributed use fixed wrt sample count estimate)
* in_chans !=3 support for scripts / loader
* Adan optimizer
* Can enable per-step LR scheduling via args
* Dataset 'parsers' renamed to 'readers', more descriptive of purpose
* AMP args changed, APEX via `--amp-impl apex`, bfloat16 supportedf via `--amp-dtype bfloat16`
* main branch switched to 0.7.x version, 0.6x forked for stable release of weight only adds
* master -> main branch rename

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
* All runtime benchmark and validation result csv files are up-to-date!
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

