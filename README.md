# PyTorch Image Models
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

## What's New

❗Updates after Oct 10, 2022 are available in version >= 0.9❗
* Many changes since the last 0.6.x stable releases. They were previewed in 0.8.x dev releases but not everyone transitioned.
* `timm.models.layers` moved to `timm.layers`:
  * `from timm.models.layers import name` will still work via deprecation mapping (but please transition to `timm.layers`).
  * `import timm.models.layers.module` or `from timm.models.layers.module import name` needs to be changed now.
* Builder, helper, non-model modules in `timm.models` have a `_` prefix added, ie `timm.models.helpers` -> `timm.models._helpers`, there are temporary deprecation mapping files but those will be removed.
* All models now support `architecture.pretrained_tag` naming (ex `resnet50.rsb_a1`).
  * The pretrained_tag is the specific weight variant (different head) for the architecture.
  * Using only `architecture` defaults to the first weights in the default_cfgs for that model architecture.
  * In adding pretrained tags, many model names that existed to differentiate were renamed to use the tag  (ex: `vit_base_patch16_224_in21k` -> `vit_base_patch16_224.augreg_in21k`). There are deprecation mappings for these.
* A number of models had their checkpoints remaped to match architecture changes needed to better support `features_only=True`, there are `checkpoint_filter_fn` methods in any model module that was remapped. These can be passed to `timm.models.load_checkpoint(..., filter_fn=timm.models.swin_transformer_v2.checkpoint_filter_fn)` to remap your existing checkpoint.
* The Hugging Face Hub (https://huggingface.co/timm) is now the primary source for `timm` weights. Model cards include link to papers, original source, license. 
* Previous 0.6.x can be cloned from [0.6.x](https://github.com/rwightman/pytorch-image-models/tree/0.6.x) branch or installed via pip with version.

### May 14, 2024
* Support loading PaliGemma jax weights into SigLIP ViT models with average pooling.
* Add Hiera models from Meta (https://github.com/facebookresearch/hiera).
* Add `normalize=` flag for transorms, return non-normalized torch.Tensor with original dytpe (for `chug`)

### May 11, 2024
* `Searching for Better ViT Baselines (For the GPU Poor)` weights and vit variants released. Exploring model shapes between Tiny and Base.

| model | top1 | top5 | param_count | img_size |
| -------------------------------------------------- | ------ | ------ | ----------- | -------- |
| [vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k](https://huggingface.co/timm/vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k) | 86.202 | 97.874 | 64.11 | 256 |
| [vit_betwixt_patch16_reg4_gap_256.sbb_in12k_ft_in1k](https://huggingface.co/timm/vit_betwixt_patch16_reg4_gap_256.sbb_in12k_ft_in1k)  | 85.418 | 97.48 | 60.4 | 256 |
| [vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k)  | 84.322 | 96.812 | 63.95 | 256 |
| [vit_betwixt_patch16_rope_reg4_gap_256.sbb_in1k](https://huggingface.co/timm/vit_betwixt_patch16_rope_reg4_gap_256.sbb_in1k)  | 83.906 | 96.684 | 60.23 | 256 |
| [vit_base_patch16_rope_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k)  | 83.866 | 96.67 | 86.43 | 256 |
| [vit_medium_patch16_rope_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_medium_patch16_rope_reg1_gap_256.sbb_in1k)  | 83.81 | 96.824 | 38.74 | 256 |
| [vit_betwixt_patch16_reg4_gap_256.sbb_in1k](https://huggingface.co/timm/vit_betwixt_patch16_reg4_gap_256.sbb_in1k)  | 83.706 | 96.616 | 60.4 | 256 |
| [vit_betwixt_patch16_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_betwixt_patch16_reg1_gap_256.sbb_in1k)  | 83.628 | 96.544 | 60.4 | 256 |
| [vit_medium_patch16_reg4_gap_256.sbb_in1k](https://huggingface.co/timm/vit_medium_patch16_reg4_gap_256.sbb_in1k)  | 83.47 | 96.622 | 38.88 | 256 |
| [vit_medium_patch16_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_medium_patch16_reg1_gap_256.sbb_in1k)  | 83.462 | 96.548 | 38.88 | 256 |
| [vit_little_patch16_reg4_gap_256.sbb_in1k](https://huggingface.co/timm/vit_little_patch16_reg4_gap_256.sbb_in1k)  | 82.514 | 96.262 | 22.52 | 256 |
| [vit_wee_patch16_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_wee_patch16_reg1_gap_256.sbb_in1k)  | 80.256 | 95.360 | 13.42 | 256 |
| [vit_pwee_patch16_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_pwee_patch16_reg1_gap_256.sbb_in1k)  | 80.072 | 95.136 | 15.25 | 256 |
| [vit_mediumd_patch16_reg4_gap_256.sbb_in12k](https://huggingface.co/timm/vit_mediumd_patch16_reg4_gap_256.sbb_in12k) | N/A | N/A | 64.11 | 256 |
| [vit_betwixt_patch16_reg4_gap_256.sbb_in12k](https://huggingface.co/timm/vit_betwixt_patch16_reg4_gap_256.sbb_in12k)  | N/A | N/A | 60.4 | 256 |

* AttentionExtract helper added to extract attention maps from `timm` models. See example in https://github.com/huggingface/pytorch-image-models/discussions/1232#discussioncomment-9320949
* `forward_intermediates()` API refined and added to more models including some ConvNets that have other extraction methods.
* 1017 of 1047 model architectures support `features_only=True` feature extraction. Remaining 34 architectures can be supported but based on priority requests.
* Remove torch.jit.script annotated functions including old JIT activations. Conflict with dynamo and dynamo does a much better job when used.

### April 11, 2024
* Prepping for a long overdue 1.0 release, things have been stable for a while now.
* Significant feature that's been missing for a while, `features_only=True` support for ViT models with flat hidden states or non-std module layouts (so far covering  `'vit_*', 'twins_*', 'deit*', 'beit*', 'mvitv2*', 'eva*', 'samvit_*', 'flexivit*'`)
* Above feature support achieved through a new `forward_intermediates()` API that can be used with a feature wrapping module or direclty.
```python
model = timm.create_model('vit_base_patch16_224')
final_feat, intermediates = model.forward_intermediates(input) 
output = model.forward_head(final_feat)  # pooling + classifier head

print(final_feat.shape)
torch.Size([2, 197, 768])

for f in intermediates:
    print(f.shape)
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])
torch.Size([2, 768, 14, 14])

print(output.shape)
torch.Size([2, 1000])
```

```python
model = timm.create_model('eva02_base_patch16_clip_224', pretrained=True, img_size=512, features_only=True, out_indices=(-3, -2,))
output = model(torch.randn(2, 3, 512, 512))

for o in output:    
    print(o.shape)   
torch.Size([2, 768, 32, 32])
torch.Size([2, 768, 32, 32])
```
* TinyCLIP vision tower weights added, thx [Thien Tran](https://github.com/gau-nernst)

### Feb 19, 2024
* Next-ViT models added. Adapted from https://github.com/bytedance/Next-ViT
* HGNet and PP-HGNetV2 models added. Adapted from https://github.com/PaddlePaddle/PaddleClas by [SeeFun](https://github.com/seefun)
* Removed setup.py, moved to pyproject.toml based build supported by PDM
* Add updated model EMA impl using _for_each for less overhead
* Support device args in train script for non GPU devices
* Other misc fixes and small additions
* Min supported Python version increased to 3.8
* Release 0.9.16

### Jan 8, 2024
Datasets & transform refactoring
* HuggingFace streaming (iterable) dataset support (`--dataset hfids:org/dataset`)
* Webdataset wrapper tweaks for improved split info fetching, can auto fetch splits from supported HF hub webdataset
* Tested HF `datasets` and webdataset wrapper streaming from HF hub with recent `timm` ImageNet uploads to https://huggingface.co/timm
* Make input & target column/field keys consistent across datasets and pass via args
* Full monochrome support when using e:g: `--input-size 1 224 224` or `--in-chans 1`, sets PIL image conversion appropriately in dataset
* Improved several alternate crop & resize transforms (ResizeKeepRatio, RandomCropOrPad, etc) for use in PixParse document AI project
* Add SimCLR style color jitter prob along with grayscale and gaussian blur options to augmentations and args
* Allow train without validation set (`--val-split ''`) in train script
* Add `--bce-sum` (sum over class dim) and `--bce-pos-weight` (positive weighting) args for training as they're common BCE loss tweaks I was often hard coding 

### Nov 23, 2023
* Added EfficientViT-Large models, thanks [SeeFun](https://github.com/seefun)
* Fix Python 3.7 compat, will be dropping support for it soon
* Other misc fixes
* Release 0.9.12

### Nov 20, 2023
* Added significant flexibility for Hugging Face Hub based timm models via `model_args` config entry. `model_args` will be passed as kwargs through to models on creation. 
  * See example at https://huggingface.co/gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k/blob/main/config.json
  * Usage: https://github.com/huggingface/pytorch-image-models/discussions/2035
* Updated imagenet eval and test set csv files with latest models
* `vision_transformer.py` typing and doc cleanup by [Laureηt](https://github.com/Laurent2916)
* 0.9.11 release

### Nov 3, 2023
* [DFN (Data Filtering Networks)](https://huggingface.co/papers/2309.17425) and [MetaCLIP](https://huggingface.co/papers/2309.16671) ViT weights added
* DINOv2 'register' ViT model weights added (https://huggingface.co/papers/2309.16588, https://huggingface.co/papers/2304.07193)
* Add `quickgelu` ViT variants for OpenAI, DFN, MetaCLIP weights that use it (less efficient)
* Improved typing added to ResNet, MobileNet-v3 thanks to [Aryan](https://github.com/a-r-r-o-w)
* ImageNet-12k fine-tuned (from LAION-2B CLIP) `convnext_xxlarge`
* 0.9.9 release

### Oct 20, 2023
* [SigLIP](https://huggingface.co/papers/2303.15343) image tower weights supported in `vision_transformer.py`.
  * Great potential for fine-tune and downstream feature use.
* Experimental 'register' support in vit models as per [Vision Transformers Need Registers](https://huggingface.co/papers/2309.16588)
* Updated RepViT with new weight release. Thanks [wangao](https://github.com/jameslahm)
* Add patch resizing support (on pretrained weight load) to Swin models
* 0.9.8 release pending

### Sep 1, 2023
* TinyViT added by [SeeFun](https://github.com/seefun)
* Fix EfficientViT (MIT) to use torch.autocast so it works back to PT 1.10
* 0.9.7 release

### Aug 28, 2023
* Add dynamic img size support to models in `vision_transformer.py`, `vision_transformer_hybrid.py`, `deit.py`, and `eva.py` w/o breaking backward compat.
  * Add `dynamic_img_size=True` to args at model creation time to allow changing the grid size (interpolate abs and/or ROPE pos embed each forward pass).
  * Add `dynamic_img_pad=True` to allow image sizes that aren't divisible by patch size (pad bottom right to patch size each forward pass).
  * Enabling either dynamic mode will break FX tracing unless PatchEmbed module added as leaf.
  * Existing method of resizing position embedding by passing different `img_size` (interpolate pretrained embed weights once) on creation still works.
  * Existing method of changing `patch_size` (resize pretrained patch_embed weights once) on creation still works.
  * Example validation cmd `python validate.py /imagenet --model vit_base_patch16_224 --amp --amp-dtype bfloat16 --img-size 255 --crop-pct 1.0 --model-kwargs dynamic_img_size=True dyamic_img_pad=True`

### Aug 25, 2023
* Many new models since last release
  * FastViT - https://arxiv.org/abs/2303.14189
  * MobileOne - https://arxiv.org/abs/2206.04040
  * InceptionNeXt - https://arxiv.org/abs/2303.16900
  * RepGhostNet - https://arxiv.org/abs/2211.06088 (thanks https://github.com/ChengpengChen)
  * GhostNetV2 - https://arxiv.org/abs/2211.12905 (thanks https://github.com/yehuitang)
  * EfficientViT (MSRA) - https://arxiv.org/abs/2305.07027 (thanks https://github.com/seefun)
  * EfficientViT (MIT) - https://arxiv.org/abs/2205.14756 (thanks https://github.com/seefun)
* Add `--reparam` arg to `benchmark.py`, `onnx_export.py`, and `validate.py` to trigger layer reparameterization / fusion for models with any one of `reparameterize()`, `switch_to_deploy()` or `fuse()`
  * Including FastViT, MobileOne, RepGhostNet, EfficientViT (MSRA), RepViT, RepVGG, and LeViT
* Preparing 0.9.6 'back to school' release

### Aug 11, 2023
* Swin, MaxViT, CoAtNet, and BEiT models support resizing of image/window size on creation with adaptation of pretrained weights
* Example validation cmd to test w/ non-square resize `python validate.py /imagenet --model swin_base_patch4_window7_224.ms_in22k_ft_in1k --amp --amp-dtype bfloat16 --input-size 3 256 320 --model-kwargs window_size=8,10 img_size=256,320`
 
### Aug 3, 2023
* Add GluonCV weights for HRNet w18_small and w18_small_v2. Converted by [SeeFun](https://github.com/seefun)
* Fix `selecsls*` model naming regression
* Patch and position embedding for ViT/EVA works for bfloat16/float16 weights on load (or activations for on-the-fly resize)
* v0.9.5 release prep

### July 27, 2023
* Added timm trained `seresnextaa201d_32x8d.sw_in12k_ft_in1k_384` weights (and `.sw_in12k` pretrain) with 87.3% top-1 on ImageNet-1k, best ImageNet ResNet family model I'm aware of.
* RepViT model and weights (https://arxiv.org/abs/2307.09283) added by [wangao](https://github.com/jameslahm)
* I-JEPA ViT feature weights (no classifier) added by [SeeFun](https://github.com/seefun)
* SAM-ViT (segment anything) feature weights (no classifier) added by [SeeFun](https://github.com/seefun)
* Add support for alternative feat extraction methods and -ve indices to EfficientNet
* Add NAdamW optimizer
* Misc fixes

### May 11, 2023
* `timm` 0.9 released, transition from 0.8.xdev releases

### May 10, 2023
* Hugging Face Hub downloading is now default, 1132 models on https://huggingface.co/timm, 1163 weights in `timm`
* DINOv2 vit feature backbone weights added thanks to [Leng Yue](https://github.com/leng-yue)
* FB MAE vit feature backbone weights added
* OpenCLIP DataComp-XL L/14 feat backbone weights added
* MetaFormer (poolformer-v2, caformer, convformer, updated poolformer (v1)) w/ weights added by [Fredo Guan](https://github.com/fffffgggg54)
* Experimental `get_intermediate_layers` function on vit/deit models for grabbing hidden states (inspired by DINO impl). This is WIP and may change significantly... feedback welcome.
* Model creation throws error if `pretrained=True` and no weights exist (instead of continuing with random initialization)
* Fix regression with inception / nasnet TF sourced weights with 1001 classes in original classifiers
* bitsandbytes (https://github.com/TimDettmers/bitsandbytes) optimizers added to factory, use `bnb` prefix, ie `bnbadam8bit`
* Misc cleanup and fixes
* Final testing before switching to a 0.9 and bringing `timm` out of pre-release state

### April 27, 2023
* 97% of `timm` models uploaded to HF Hub and almost all updated to support multi-weight pretrained configs
* Minor cleanup and refactoring of another batch of models as multi-weight added. More fused_attn (F.sdpa) and features_only support, and torchscript fixes.

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

## Introduction

Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arxiv papers, etc in the README, documentation, and code docstrings. Please let me know if I missed anything.

## Features

### Models

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
* EfficientViT (MIT) - https://arxiv.org/abs/2205.14756
* EfficientViT (MSRA) - https://arxiv.org/abs/2305.07027
* EVA - https://arxiv.org/abs/2211.07636
* EVA-02 - https://arxiv.org/abs/2303.11331
* FastViT - https://arxiv.org/abs/2303.14189
* FlexiViT - https://arxiv.org/abs/2212.08013
* FocalNet (Focal Modulation Networks) - https://arxiv.org/abs/2203.11926
* GCViT (Global Context Vision Transformer) - https://arxiv.org/abs/2206.09959
* GhostNet - https://arxiv.org/abs/1911.11907
* GhostNet-V2 - https://arxiv.org/abs/2211.12905
* gMLP - https://arxiv.org/abs/2105.08050
* GPU-Efficient Networks - https://arxiv.org/abs/2006.14090
* Halo Nets - https://arxiv.org/abs/2103.12731
* HGNet / HGNet-V2 - TBD
* HRNet - https://arxiv.org/abs/1908.07919
* InceptionNeXt - https://arxiv.org/abs/2303.16900
* Inception-V3 - https://arxiv.org/abs/1512.00567
* Inception-ResNet-V2 and Inception-V4 - https://arxiv.org/abs/1602.07261
* Lambda Networks - https://arxiv.org/abs/2102.08602
* LeViT (Vision Transformer in ConvNet's Clothing) - https://arxiv.org/abs/2104.01136
* MaxViT (Multi-Axis Vision Transformer) - https://arxiv.org/abs/2204.01697
* MetaFormer (PoolFormer-v2, ConvFormer, CAFormer) - https://arxiv.org/abs/2210.13452
* MLP-Mixer - https://arxiv.org/abs/2105.01601
* MobileNet-V3 (MBConvNet w/ Efficient Head) - https://arxiv.org/abs/1905.02244
  * FBNet-V3 - https://arxiv.org/abs/2006.02049
  * HardCoRe-NAS - https://arxiv.org/abs/2102.11646
  * LCNet - https://arxiv.org/abs/2109.15099
* MobileOne - https://arxiv.org/abs/2206.04040
* MobileViT - https://arxiv.org/abs/2110.02178
* MobileViT-V2 - https://arxiv.org/abs/2206.02680
* MViT-V2 (Improved Multiscale Vision Transformer) - https://arxiv.org/abs/2112.01526
* NASNet-A - https://arxiv.org/abs/1707.07012
* NesT - https://arxiv.org/abs/2105.12723
* Next-ViT - https://arxiv.org/abs/2207.05501
* NFNet-F - https://arxiv.org/abs/2102.06171
* NF-RegNet / NF-ResNet - https://arxiv.org/abs/2101.08692
* PNasNet - https://arxiv.org/abs/1712.00559
* PoolFormer (MetaFormer) - https://arxiv.org/abs/2111.11418
* Pooling-based Vision Transformer (PiT) - https://arxiv.org/abs/2103.16302
* PVT-V2 (Improved Pyramid Vision Transformer) - https://arxiv.org/abs/2106.13797
* RegNet - https://arxiv.org/abs/2003.13678
* RegNetZ - https://arxiv.org/abs/2103.06877
* RepVGG - https://arxiv.org/abs/2101.03697
* RepGhostNet - https://arxiv.org/abs/2211.06088
* RepViT - https://arxiv.org/abs/2307.09283
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

### Optimizers

Included optimizers available via `create_optimizer` / `create_optimizer_v2` factory methods:
* `adabelief` an implementation of AdaBelief adapted from https://github.com/juntang-zhuang/Adabelief-Optimizer - https://arxiv.org/abs/2010.07468
* `adafactor` adapted from [FAIRSeq impl](https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py) - https://arxiv.org/abs/1804.04235
* `adahessian` by [David Samuel](https://github.com/davda54/ada-hessian) - https://arxiv.org/abs/2006.00719
* `adamp` and `sgdp` by [Naver ClovAI](https://github.com/clovaai) - https://arxiv.org/abs/2006.08217
* `adan` an implementation of Adan adapted from https://github.com/sail-sg/Adan - https://arxiv.org/abs/2208.06677
* `lamb` an implementation of Lamb and LambC (w/ trust-clipping) cleaned up and modified to support use with XLA - https://arxiv.org/abs/1904.00962
* `lars` an implementation of LARS and LARC (w/ trust-clipping) - https://arxiv.org/abs/1708.03888
* `lion` and implementation of Lion adapted from https://github.com/google/automl/tree/master/lion - https://arxiv.org/abs/2302.06675
* `lookahead` adapted from impl by [Liam](https://github.com/alphadl/lookahead.pytorch) - https://arxiv.org/abs/1907.08610
* `madgrad` - and implementation of MADGRAD adapted from https://github.com/facebookresearch/madgrad - https://arxiv.org/abs/2101.11075
* `nadam` an implementation of Adam w/ Nesterov momentum
* `nadamw` an impementation of AdamW (Adam w/ decoupled weight-decay) w/ Nesterov momentum. A simplified impl based on https://github.com/mlcommons/algorithmic-efficiency
* `novograd` by [Masashi Kimura](https://github.com/convergence-lab/novograd) - https://arxiv.org/abs/1905.11286
* `radam` by [Liyuan Liu](https://github.com/LiyuanLucasLiu/RAdam) - https://arxiv.org/abs/1908.03265
* `rmsprop_tf` adapted from PyTorch RMSProp by myself. Reproduces much improved Tensorflow RMSProp behaviour
* `sgdw` and implementation of SGD w/ decoupled weight-decay
* `fused<name>` optimizers by name with [NVIDIA Apex](https://github.com/NVIDIA/apex/tree/master/apex/optimizers) installed
* `bits<name>` optimizers by name with [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) installed

### Augmentations
* Random Erasing from [Zhun Zhong](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py) - https://arxiv.org/abs/1708.04896)
* Mixup - https://arxiv.org/abs/1710.09412
* CutMix - https://arxiv.org/abs/1905.04899
* AutoAugment (https://arxiv.org/abs/1805.09501) and RandAugment (https://arxiv.org/abs/1909.13719) ImageNet configurations modeled after impl for EfficientNet training (https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py)
* AugMix w/ JSD loss, JSD w/ clean + augmented mixing support works with AutoAugment and RandAugment as well - https://arxiv.org/abs/1912.02781
* SplitBachNorm - allows splitting batch norm layers between clean and augmented (auxiliary batch norm) data

### Regularization
* DropPath aka "Stochastic Depth" - https://arxiv.org/abs/1603.09382
* DropBlock - https://arxiv.org/abs/1810.12890
* Blur Pooling - https://arxiv.org/abs/1904.11486

### Other

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
