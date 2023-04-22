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

* ❗Updates after Oct 10, 2022 are available in 0.8.x pre-releases (`pip install --pre timm`) or cloning main❗
* Stable releases are 0.6.x and available by normal pip install or clone from [0.6.x](https://github.com/rwightman/pytorch-image-models/tree/0.6.x) branch.

### April 21, 2023
* Gradient accumulation support added to train script and tested (`--grad-accum-steps`), thanks [Taeksang Kim](https://github.com/voidbag)
* More weights on HF Hub (cspnet, cait, volo, xcit, tresnet, hardcorenas, densenet, dpn, vovnet, xception_aligned)
* Added `--head-init-scale` and `--head-init-bias` to train.py to scale classiifer head and set fixed bias for fine-tune
* Remove all InplaceABN (`inplace_abn`) use, replaced use in tresnet with standard BatchNorm (modified weights accordingly). 

### April 12, 2023
* Add ONNX export script, validate script, helpers that I've had kicking around for along time. Tweak 'same' padding for better export w/ recent ONNX + pytorch.
* Refactor dropout args for vit and vit-like models, separate drop_rate into `drop_rate` (classifier dropout), `proj_drop_rate` (block mlp / out projections), `pos_drop_rate` (position embedding drop), `attn_drop_rate` (attention dropout). Also add patch dropout (FLIP) to vit and eva models.
* fused F.scaled_dot_product_attention support to more vit models, add env var (TIMM_FUSED_ATTN) to control, and config interface to enable/disable
* Add EVA-CLIP backbones w/ image tower weights, all the way up to 4B param 'enormous' model, and 336x336 OpenAI ViT mode that was missed.

### April 5, 2023
* ALL ResNet models pushed to Hugging Face Hub with multi-weight support
  * All past `timm` trained weights added with recipe based tags to differentiate
  * All ResNet strikes back A1/A2/A3 (seed 0) and R50 example B/C1/C2/D weights available
  * Add torchvision v2 recipe weights to existing torchvision originals
  * See comparison table in https://huggingface.co/timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288#model-comparison
* New ImageNet-12k + ImageNet-1k fine-tunes available for a few anti-aliased ResNet models
  * `resnetaa50d.sw_in12k_ft_in1k` - 81.7 @ 224, 82.6 @ 288
  * `resnetaa101d.sw_in12k_ft_in1k` - 83.5 @ 224, 84.1 @ 288
  * `seresnextaa101d_32x8d.sw_in12k_ft_in1k` - 86.0 @ 224, 86.5 @ 288 
  * `seresnextaa101d_32x8d.sw_in12k_ft_in1k_288` - 86.5 @ 288, 86.7 @ 320

### March 31, 2023
* Add first ConvNext-XXLarge CLIP -> IN-1k fine-tune and IN-12k intermediate fine-tunes for convnext-base/large CLIP models.

| model                                                                                                                |top1  |top5  |img_size|param_count|gmacs |macts |
|----------------------------------------------------------------------------------------------------------------------|------|------|--------|-----------|------|------|
| [convnext_xxlarge.clip_laion2b_soup_ft_in1k](https://huggingface.co/timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k) |88.612|98.704|256     |846.47     |198.09|124.45|
| convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384                                                               |88.312|98.578|384     |200.13     |101.11|126.74|
| convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320                                                               |87.968|98.47 |320     |200.13     |70.21 |88.02 |
| convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384                                                                  |87.138|98.212|384     |88.59      |45.21 |84.49 |
| convnext_base.clip_laion2b_augreg_ft_in12k_in1k                                                                      |86.344|97.97 |256     |88.59      |20.09 |37.55 |

* Add EVA-02 MIM pretrained and fine-tuned weights, push to HF hub and update model cards for all EVA models. First model over 90% top-1 (99% top-5)! Check out the original code & weights at https://github.com/baaivision/EVA for more details on their work blending MIM, CLIP w/ many model, dataset, and train recipe tweaks.

| model                                              |top1  |top5  |param_count|img_size|
|----------------------------------------------------|------|------|-----------|--------|
| [eva02_large_patch14_448.mim_m38m_ft_in22k_in1k](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in1k) |90.054|99.042|305.08     |448     |
| eva02_large_patch14_448.mim_in22k_ft_in22k_in1k    |89.946|99.01 |305.08     |448     |
| eva_giant_patch14_560.m30m_ft_in22k_in1k           |89.792|98.992|1014.45    |560     |
| eva02_large_patch14_448.mim_in22k_ft_in1k          |89.626|98.954|305.08     |448     |
| eva02_large_patch14_448.mim_m38m_ft_in1k           |89.57 |98.918|305.08     |448     |
| eva_giant_patch14_336.m30m_ft_in22k_in1k           |89.56 |98.956|1013.01    |336     |
| eva_giant_patch14_336.clip_ft_in1k                 |89.466|98.82 |1013.01    |336     |
| eva_large_patch14_336.in22k_ft_in22k_in1k          |89.214|98.854|304.53     |336     |
| eva_giant_patch14_224.clip_ft_in1k                 |88.882|98.678|1012.56    |224     |
| eva02_base_patch14_448.mim_in22k_ft_in22k_in1k     |88.692|98.722|87.12      |448     |
| eva_large_patch14_336.in22k_ft_in1k                |88.652|98.722|304.53     |336     |
| eva_large_patch14_196.in22k_ft_in22k_in1k          |88.592|98.656|304.14     |196     |
| eva02_base_patch14_448.mim_in22k_ft_in1k           |88.23 |98.564|87.12      |448     |
| eva_large_patch14_196.in22k_ft_in1k                |87.934|98.504|304.14     |196     |
| eva02_small_patch14_336.mim_in22k_ft_in1k          |85.74 |97.614|22.13      |336     |
| eva02_tiny_patch14_336.mim_in22k_ft_in1k           |80.658|95.524|5.76       |336     |

* Multi-weight and HF hub for DeiT and MLP-Mixer based models

### March 22, 2023
* More weights pushed to HF hub along with multi-weight support, including: `regnet.py`, `rexnet.py`, `byobnet.py`, `resnetv2.py`, `swin_transformer.py`, `swin_transformer_v2.py`, `swin_transformer_v2_cr.py`
* Swin Transformer models support feature extraction (NCHW feat maps for `swinv2_cr_*`, and NHWC for all others) and spatial embedding outputs.
* FocalNet (from https://github.com/microsoft/FocalNet) models and weights added with significant refactoring, feature extraction, no fixed resolution / sizing constraint
* RegNet weights increased with HF hub push, SWAG, SEER, and torchvision v2 weights. SEER is pretty poor wrt to performance for model size, but possibly useful.
* More ImageNet-12k pretrained and 1k fine-tuned `timm` weights:
  * `rexnetr_200.sw_in12k_ft_in1k` - 82.6 @ 224, 83.2 @ 288
  * `rexnetr_300.sw_in12k_ft_in1k` - 84.0 @ 224, 84.5 @ 288
  * `regnety_120.sw_in12k_ft_in1k` - 85.0 @ 224, 85.4 @ 288
  * `regnety_160.lion_in12k_ft_in1k` - 85.6 @ 224, 86.0 @ 288
  * `regnety_160.sw_in12k_ft_in1k` - 85.6 @ 224, 86.0 @ 288  (compare to SWAG PT + 1k FT this is same BUT much lower res, blows SEER FT away)
* Model name deprecation + remapping functionality added (a milestone for bringing 0.8.x out of pre-release). Mappings being added...
* Minor bug fixes and improvements.

### Feb 26, 2023
* Add ConvNeXt-XXLarge CLIP pretrained image tower weights for fine-tune & features (fine-tuning TBD) -- see [model card](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)
* Update `convnext_xxlarge` default LayerNorm eps to 1e-5 (for CLIP weights, improved stability)
* 0.8.15dev0

### Feb 20, 2023
* Add 320x320 `convnext_large_mlp.clip_laion2b_ft_320` and `convnext_lage_mlp.clip_laion2b_ft_soup_320` CLIP image tower weights for features & fine-tune
* 0.8.13dev0 pypi release for latest changes w/ move to huggingface org

### Feb 16, 2023
* `safetensor` checkpoint support added
* Add ideas from 'Scaling Vision Transformers to 22 B. Params' (https://arxiv.org/abs/2302.05442) -- qk norm, RmsNorm, parallel block
* Add F.scaled_dot_product_attention support (PyTorch 2.0 only) to `vit_*`, `vit_relpos*`, `coatnet` / `maxxvit` (to start)
* Lion optimizer (w/ multi-tensor option) added (https://arxiv.org/abs/2302.06675)
* gradient checkpointing works with `features_only=True`

### Feb 7, 2023
* New inference benchmark numbers added in [results](results/) folder.
* Add convnext LAION CLIP trained weights and initial set of in1k fine-tunes
  * `convnext_base.clip_laion2b_augreg_ft_in1k` - 86.2% @ 256x256
  * `convnext_base.clip_laiona_augreg_ft_in1k_384` - 86.5% @ 384x384
  * `convnext_large_mlp.clip_laion2b_augreg_ft_in1k` - 87.3% @ 256x256
  * `convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384` - 87.9% @ 384x384
* Add DaViT models. Supports `features_only=True`. Adapted from https://github.com/dingmyu/davit by [Fredo](https://github.com/fffffgggg54).
* Use a common NormMlpClassifierHead across MaxViT, ConvNeXt, DaViT
* Add EfficientFormer-V2 model, update EfficientFormer, and refactor LeViT (closely related architectures). Weights on HF hub.
  * New EfficientFormer-V2 arch, significant refactor from original at (https://github.com/snap-research/EfficientFormer). Supports `features_only=True`.
  * Minor updates to EfficientFormer.
  * Refactor LeViT models to stages, add `features_only=True` support to new `conv` variants, weight remap required.
* Move ImageNet meta-data (synsets, indices) from `/results` to [`timm/data/_info`](timm/data/_info/).
* Add ImageNetInfo / DatasetInfo classes to provide labelling for various ImageNet classifier layouts in `timm`
  * Update `inference.py` to use, try: `python inference.py /folder/to/images --model convnext_small.in12k --label-type detail --topk 5`
* Ready for 0.8.10 pypi pre-release (final testing).

### Jan 20, 2023
* Add two convnext 12k -> 1k fine-tunes at 384x384
  * `convnext_tiny.in12k_ft_in1k_384` - 85.1 @ 384
  * `convnext_small.in12k_ft_in1k_384` - 86.2 @ 384

* Push all MaxxViT weights to HF hub, and add new ImageNet-12k -> 1k fine-tunes for `rw` base MaxViT and CoAtNet 1/2 models

|model                                                                                                                   |top1 |top5 |samples / sec  |Params (M)     |GMAC  |Act (M)|
|------------------------------------------------------------------------------------------------------------------------|----:|----:|--------------:|--------------:|-----:|------:|
|[maxvit_xlarge_tf_512.in21k_ft_in1k](https://huggingface.co/timm/maxvit_xlarge_tf_512.in21k_ft_in1k)                    |88.53|98.64|          21.76|         475.77|534.14|1413.22|
|[maxvit_xlarge_tf_384.in21k_ft_in1k](https://huggingface.co/timm/maxvit_xlarge_tf_384.in21k_ft_in1k)                    |88.32|98.54|          42.53|         475.32|292.78| 668.76|
|[maxvit_base_tf_512.in21k_ft_in1k](https://huggingface.co/timm/maxvit_base_tf_512.in21k_ft_in1k)                        |88.20|98.53|          50.87|         119.88|138.02| 703.99|
|[maxvit_large_tf_512.in21k_ft_in1k](https://huggingface.co/timm/maxvit_large_tf_512.in21k_ft_in1k)                      |88.04|98.40|          36.42|         212.33|244.75| 942.15|
|[maxvit_large_tf_384.in21k_ft_in1k](https://huggingface.co/timm/maxvit_large_tf_384.in21k_ft_in1k)                      |87.98|98.56|          71.75|         212.03|132.55| 445.84|
|[maxvit_base_tf_384.in21k_ft_in1k](https://huggingface.co/timm/maxvit_base_tf_384.in21k_ft_in1k)                        |87.92|98.54|         104.71|         119.65| 73.80| 332.90|
|[maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k](https://huggingface.co/timm/maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k)        |87.81|98.37|         106.55|         116.14| 70.97| 318.95|
|[maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k](https://huggingface.co/timm/maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k)  |87.47|98.37|         149.49|         116.09| 72.98| 213.74|
|[coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k)            |87.39|98.31|         160.80|          73.88| 47.69| 209.43|
|[maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k](https://huggingface.co/timm/maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k)        |86.89|98.02|         375.86|         116.14| 23.15|  92.64|
|[maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k](https://huggingface.co/timm/maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k)  |86.64|98.02|         501.03|         116.09| 24.20|  62.77|
|[maxvit_base_tf_512.in1k](https://huggingface.co/timm/maxvit_base_tf_512.in1k)                                          |86.60|97.92|          50.75|         119.88|138.02| 703.99|
|[coatnet_2_rw_224.sw_in12k_ft_in1k](https://huggingface.co/timm/coatnet_2_rw_224.sw_in12k_ft_in1k)                      |86.57|97.89|         631.88|          73.87| 15.09|  49.22|
|[maxvit_large_tf_512.in1k](https://huggingface.co/timm/maxvit_large_tf_512.in1k)                                        |86.52|97.88|          36.04|         212.33|244.75| 942.15|
|[coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k](https://huggingface.co/timm/coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k)            |86.49|97.90|         620.58|          73.88| 15.18|  54.78|
|[maxvit_base_tf_384.in1k](https://huggingface.co/timm/maxvit_base_tf_384.in1k)                                          |86.29|97.80|         101.09|         119.65| 73.80| 332.90|
|[maxvit_large_tf_384.in1k](https://huggingface.co/timm/maxvit_large_tf_384.in1k)                                        |86.23|97.69|          70.56|         212.03|132.55| 445.84|
|[maxvit_small_tf_512.in1k](https://huggingface.co/timm/maxvit_small_tf_512.in1k)                                        |86.10|97.76|          88.63|          69.13| 67.26| 383.77|
|[maxvit_tiny_tf_512.in1k](https://huggingface.co/timm/maxvit_tiny_tf_512.in1k)                                          |85.67|97.58|         144.25|          31.05| 33.49| 257.59|
|[maxvit_small_tf_384.in1k](https://huggingface.co/timm/maxvit_small_tf_384.in1k)                                        |85.54|97.46|         188.35|          69.02| 35.87| 183.65|
|[maxvit_tiny_tf_384.in1k](https://huggingface.co/timm/maxvit_tiny_tf_384.in1k)                                          |85.11|97.38|         293.46|          30.98| 17.53| 123.42|
|[maxvit_large_tf_224.in1k](https://huggingface.co/timm/maxvit_large_tf_224.in1k)                                        |84.93|96.97|         247.71|         211.79| 43.68| 127.35|
|[coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k](https://huggingface.co/timm/coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k)          |84.90|96.96|        1025.45|          41.72|  8.11|  40.13|
|[maxvit_base_tf_224.in1k](https://huggingface.co/timm/maxvit_base_tf_224.in1k)                                          |84.85|96.99|         358.25|         119.47| 24.04|  95.01|
|[maxxvit_rmlp_small_rw_256.sw_in1k](https://huggingface.co/timm/maxxvit_rmlp_small_rw_256.sw_in1k)                      |84.63|97.06|         575.53|          66.01| 14.67|  58.38|
|[coatnet_rmlp_2_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_rmlp_2_rw_224.sw_in1k)                              |84.61|96.74|         625.81|          73.88| 15.18|  54.78|
|[maxvit_rmlp_small_rw_224.sw_in1k](https://huggingface.co/timm/maxvit_rmlp_small_rw_224.sw_in1k)                        |84.49|96.76|         693.82|          64.90| 10.75|  49.30|
|[maxvit_small_tf_224.in1k](https://huggingface.co/timm/maxvit_small_tf_224.in1k)                                        |84.43|96.83|         647.96|          68.93| 11.66|  53.17|
|[maxvit_rmlp_tiny_rw_256.sw_in1k](https://huggingface.co/timm/maxvit_rmlp_tiny_rw_256.sw_in1k)                          |84.23|96.78|         807.21|          29.15|  6.77|  46.92|
|[coatnet_1_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_1_rw_224.sw_in1k)                                        |83.62|96.38|         989.59|          41.72|  8.04|  34.60|
|[maxvit_tiny_rw_224.sw_in1k](https://huggingface.co/timm/maxvit_tiny_rw_224.sw_in1k)                                    |83.50|96.50|        1100.53|          29.06|  5.11|  33.11|
|[maxvit_tiny_tf_224.in1k](https://huggingface.co/timm/maxvit_tiny_tf_224.in1k)                                          |83.41|96.59|        1004.94|          30.92|  5.60|  35.78|
|[coatnet_rmlp_1_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_rmlp_1_rw_224.sw_in1k)                              |83.36|96.45|        1093.03|          41.69|  7.85|  35.47|
|[maxxvitv2_nano_rw_256.sw_in1k](https://huggingface.co/timm/maxxvitv2_nano_rw_256.sw_in1k)                              |83.11|96.33|        1276.88|          23.70|  6.26|  23.05|
|[maxxvit_rmlp_nano_rw_256.sw_in1k](https://huggingface.co/timm/maxxvit_rmlp_nano_rw_256.sw_in1k)                        |83.03|96.34|        1341.24|          16.78|  4.37|  26.05|
|[maxvit_rmlp_nano_rw_256.sw_in1k](https://huggingface.co/timm/maxvit_rmlp_nano_rw_256.sw_in1k)                          |82.96|96.26|        1283.24|          15.50|  4.47|  31.92|
|[maxvit_nano_rw_256.sw_in1k](https://huggingface.co/timm/maxvit_nano_rw_256.sw_in1k)                                    |82.93|96.23|        1218.17|          15.45|  4.46|  30.28|
|[coatnet_bn_0_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_bn_0_rw_224.sw_in1k)                                  |82.39|96.19|        1600.14|          27.44|  4.67|  22.04|
|[coatnet_0_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k)                                        |82.39|95.84|        1831.21|          27.44|  4.43|  18.73|
|[coatnet_rmlp_nano_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_rmlp_nano_rw_224.sw_in1k)                        |82.05|95.87|        2109.09|          15.15|  2.62|  20.34|
|[coatnext_nano_rw_224.sw_in1k](https://huggingface.co/timm/coatnext_nano_rw_224.sw_in1k)                                |81.95|95.92|        2525.52|          14.70|  2.47|  12.80|
|[coatnet_nano_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_nano_rw_224.sw_in1k)                                  |81.70|95.64|        2344.52|          15.14|  2.41|  15.41|
|[maxvit_rmlp_pico_rw_256.sw_in1k](https://huggingface.co/timm/maxvit_rmlp_pico_rw_256.sw_in1k)                          |80.53|95.21|        1594.71|           7.52|  1.85|  24.86|

### Jan 11, 2023
* Update ConvNeXt ImageNet-12k pretrain series w/ two new fine-tuned weights (and pre FT `.in12k` tags)
  * `convnext_nano.in12k_ft_in1k` - 82.3 @ 224, 82.9 @ 288  (previously released)
  * `convnext_tiny.in12k_ft_in1k` - 84.2 @ 224, 84.5 @ 288
  * `convnext_small.in12k_ft_in1k` - 85.2 @ 224, 85.3 @ 288

### Jan 6, 2023
* Finally got around to adding `--model-kwargs` and `--opt-kwargs` to scripts to pass through rare args directly to model classes from cmd line
  * `train.py /imagenet --model resnet50 --amp --model-kwargs output_stride=16 act_layer=silu`
  * `train.py /imagenet --model vit_base_patch16_clip_224 --img-size 240 --amp --model-kwargs img_size=240 patch_size=12`
* Cleanup some popular models to better support arg passthrough / merge with model configs, more to go.

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

## Introduction

Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arxiv papers, etc in the README, documentation, and code docstrings. Please let me know if I missed anything.

## Models

All model architecture families include variants with pretrained weights. There are specific model variants without any weights, it is NOT a bug. Help training new or better weights is always appreciated.

* Aggregating Nested Transformers - https://arxiv.org/abs/2105.12723
* BEiT - https://arxiv.org/abs/2106.08254
* Big Transfer ResNetV2 (BiT) - https://arxiv.org/abs/1912.11370
* Bottleneck Transformers - https://arxiv.org/abs/2101.11605
* CaiT (Class-Attention in Image Transformers) - https://arxiv.org/abs/2103.17239
* CoaT (Co-Scale Conv-Attentional Image Transformers) - https://arxiv.org/abs/2104.06399
* CoAtNet (Convolution and Attention) - https://arxiv.org/abs/2106.04803
* ConvNeXt - https://arxiv.org/abs/2201.03545
* ConvNeXt-V2 - http://arxiv.org/abs/2301.00808
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
* EVA - https://arxiv.org/abs/2211.07636
* EVA-02 - https://arxiv.org/abs/2303.11331
* FlexiViT - https://arxiv.org/abs/2212.08013
* FocalNet (Focal Modulation Networks) - https://arxiv.org/abs/2203.11926
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
    * doing a forward pass on just the features - `forward_features` (see [documentation](https://huggingface.co/docs/timm/feature_extraction))
    * these makes it easy to write consistent network wrappers that work with any of the models
* All models support multi-scale feature map extraction (feature pyramids) via create_model (see [documentation](https://huggingface.co/docs/timm/feature_extraction))
    * `create_model(name, features_only=True, out_indices=..., output_stride=...)`
    * `out_indices` creation arg specifies which feature maps to return, these indices are 0 based and generally correspond to the `C(i + 1)` feature level.
    * `output_stride` creation arg controls output stride of the network by using dilated convolutions. Most networks are stride 32 by default. Not all networks support this.
    * feature map channel counts, reduction level (stride) can be queried AFTER model creation via the `.feature_info` member
* All models have a consistent pretrained weight loader that adapts last linear if necessary, and from 3 to 1 channel input if desired
* High performance [reference training, validation, and inference scripts](https://huggingface.co/docs/timm/training_script) that work in several process/GPU modes:
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

Model validation results can be found in the [results tables](results/README.md)

## Getting Started (Documentation)

The official documentation can be found at https://huggingface.co/docs/hub/timm. Documentation contributions are welcome.

[Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055) by [Chris Hughes](https://github.com/Chris-hughes10) is an extensive blog post covering many aspects of `timm` in detail.

[timmdocs](http://timm.fast.ai/) is an alternate set of documentation for `timm`. A big thanks to [Aman Arora](https://github.com/amaarora) for his efforts creating timmdocs.

[paperswithcode](https://paperswithcode.com/lib/timm) is a good resource for browsing the models within `timm`.

## Train, Validation, Inference Scripts

The root folder of the repository contains reference train, validation, and inference scripts that work with the included models and other features of this repository. They are adaptable for other datasets and use cases with a little hacking. See [documentation](https://huggingface.co/docs/timm/training_script).

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
