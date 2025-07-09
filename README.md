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

## July 7, 2025
* MobileNet-v5 backbone tweaks for improved Google Gemma 3n behaviour (to pair with updated official weights)
  * Add stem bias (zero'd in updated weights, compat break with old weights)
  * GELU -> GELU (tanh approx). A minor change to be closer to JAX
* Add two arguments to layer-decay support, a min scale clamp and 'no optimization' scale threshold
* Add 'Fp32' LayerNorm, RMSNorm, SimpleNorm variants that can be enabled to force computation of norm in float32
* Some typing, argument cleanup for norm, norm+act layers done with above
* Support Naver ROPE-ViT (https://github.com/naver-ai/rope-vit) in `eva.py`, add RotaryEmbeddingMixed module for mixed mode, weights on HuggingFace Hub

|model                                             |img_size|top1  |top5  |param_count|
|--------------------------------------------------|--------|------|------|-----------|
|vit_large_patch16_rope_mixed_ape_224.naver_in1k  |224     |84.84 |97.122|304.4      |
|vit_large_patch16_rope_mixed_224.naver_in1k      |224     |84.828|97.116|304.2      |
|vit_large_patch16_rope_ape_224.naver_in1k        |224     |84.65 |97.154|304.37     |
|vit_large_patch16_rope_224.naver_in1k            |224     |84.648|97.122|304.17     |
|vit_base_patch16_rope_mixed_ape_224.naver_in1k   |224     |83.894|96.754|86.59      |
|vit_base_patch16_rope_mixed_224.naver_in1k       |224     |83.804|96.712|86.44      |
|vit_base_patch16_rope_ape_224.naver_in1k         |224     |83.782|96.61 |86.59      |
|vit_base_patch16_rope_224.naver_in1k             |224     |83.718|96.672|86.43      |
|vit_small_patch16_rope_224.naver_in1k            |224     |81.23 |95.022|21.98      |
|vit_small_patch16_rope_mixed_224.naver_in1k      |224     |81.216|95.022|21.99      |
|vit_small_patch16_rope_ape_224.naver_in1k        |224     |81.004|95.016|22.06      |
|vit_small_patch16_rope_mixed_ape_224.naver_in1k  |224     |80.986|94.976|22.06      |
* Some cleanup of ROPE modules, helpers, and FX tracing leaf registration
* Preparing version 1.0.17 release

## June 26, 2025
* MobileNetV5 backbone (w/ encoder only variant) for [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n#parameters) image encoder
* Version 1.0.16 released

## June 23, 2025
* Add F.grid_sample based 2D and factorized pos embed resize to NaFlexViT. Faster when lots of different sizes (based on example by https://github.com/stas-sl).
* Further speed up patch embed resample by replacing vmap with matmul (based on snippet by https://github.com/stas-sl).
* Add 3 initial native aspect NaFlexViT checkpoints created while testing, ImageNet-1k and 3 different pos embed configs w/ same hparams.

 | Model | Top-1 Acc | Top-5 Acc | Params (M) | Eval Seq Len |
 |:---|:---:|:---:|:---:|:---:|
 | [naflexvit_base_patch16_par_gap.e300_s576_in1k](https://hf.co/timm/naflexvit_base_patch16_par_gap.e300_s576_in1k) | 83.67 | 96.45 | 86.63 | 576 |
 | [naflexvit_base_patch16_parfac_gap.e300_s576_in1k](https://hf.co/timm/naflexvit_base_patch16_parfac_gap.e300_s576_in1k) | 83.63 | 96.41 | 86.46 | 576 |
 | [naflexvit_base_patch16_gap.e300_s576_in1k](https://hf.co/timm/naflexvit_base_patch16_gap.e300_s576_in1k) | 83.50 | 96.46 | 86.63 | 576 |
* Support gradient checkpointing for `forward_intermediates` and fix some checkpointing bugs. Thanks https://github.com/brianhou0208
* Add 'corrected weight decay' (https://arxiv.org/abs/2506.02285) as option to AdamW (legacy), Adopt, Kron, Adafactor (BV), Lamb, LaProp, Lion, NadamW, RmsPropTF, SGDW optimizers
* Switch PE (perception encoder) ViT models to use native timm weights instead of remapping on the fly
* Fix cuda stream bug in prefetch loader
  
## June 5, 2025
* Initial NaFlexVit model code. NaFlexVit is a Vision Transformer with:
  1. Encapsulated embedding and position encoding in a single module
  2. Support for nn.Linear patch embedding on pre-patchified (dictionary) inputs
  3. Support for NaFlex variable aspect, variable resolution (SigLip-2: https://arxiv.org/abs/2502.14786)
  4. Support for FlexiViT variable patch size (https://arxiv.org/abs/2212.08013)
  5. Support for NaViT fractional/factorized position embedding (https://arxiv.org/abs/2307.06304)
* Existing vit models in `vision_transformer.py` can be loaded into the NaFlexVit model by adding the `use_naflex=True` flag to `create_model`
  * Some native weights coming soon
* A full NaFlex data pipeline is available that allows training / fine-tuning / evaluating with variable aspect / size images
  * To enable in `train.py` and `validate.py` add the `--naflex-loader` arg, must be used with a NaFlexVit
* To evaluate an existing (classic) ViT loaded in NaFlexVit model w/ NaFlex data pipe:
  * `python validate.py /imagenet --amp -j 8 --model vit_base_patch16_224 --model-kwargs use_naflex=True --naflex-loader --naflex-max-seq-len 256` 
* The training has some extra args features worth noting
  * The `--naflex-train-seq-lens'` argument specifies which sequence lengths to randomly pick from per batch during training
  * The `--naflex-max-seq-len` argument sets the target sequence length for validation
  * Adding `--model-kwargs enable_patch_interpolator=True --naflex-patch-sizes 12 16 24` will enable random patch size selection per-batch w/ interpolation
  * The `--naflex-loss-scale` arg changes loss scaling mode per batch relative to the batch size, `timm` NaFlex loading changes the batch size for each seq len

## May 28, 2025
* Add a number of small/fast models thanks to https://github.com/brianhou0208
  * SwiftFormer - [(ICCV2023) SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications](https://github.com/Amshaker/SwiftFormer) 
  * FasterNet - [(CVPR2023) Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks](https://github.com/JierunChen/FasterNet)
  * SHViT - [(CVPR2024) SHViT: Single-Head Vision Transformer with Memory Efficient](https://github.com/ysj9909/SHViT)
  * StarNet - [(CVPR2024) Rewrite the Stars](https://github.com/ma-xu/Rewrite-the-Stars)
  * GhostNet-V3 [GhostNetV3: Exploring the Training Strategies for Compact Models](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv3_pytorch)
* Update EVA ViT (closest match) to support Perception Encoder models (https://arxiv.org/abs/2504.13181) from Meta, loading Hub weights but I still need to push dedicated `timm` weights
  * Add some flexibility to ROPE impl
* Big increase in number of models supporting `forward_intermediates()` and some additional fixes thanks to https://github.com/brianhou0208
  * DaViT, EdgeNeXt, EfficientFormerV2, EfficientViT(MIT), EfficientViT(MSRA), FocalNet, GCViT, HGNet /V2, InceptionNeXt, Inception-V4, MambaOut, MetaFormer, NesT, Next-ViT, PiT, PVT V2, RepGhostNet, RepViT, ResNetV2, ReXNet, TinyViT, TResNet, VoV
* TNT model updated w/ new weights `forward_intermediates()` thanks to https://github.com/brianhou0208
* Add `local-dir:` pretrained schema, can use `local-dir:/path/to/model/folder` for model name to source model / pretrained cfg & weights Hugging Face Hub models (config.json + weights file) from a local folder.
* Fixes, improvements for onnx export
    
## Feb 21, 2025
* SigLIP 2 ViT image encoders added (https://huggingface.co/collections/timm/siglip-2-67b8e72ba08b09dd97aecaf9)
  * Variable resolution / aspect NaFlex versions are a WIP
* Add 'SO150M2' ViT weights trained with SBB recipes, great results, better for ImageNet than previous attempt w/ less training.
  * `vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k` - 88.1% top-1
  * `vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k` - 87.9% top-1
  * `vit_so150m2_patch16_reg1_gap_256.sbb_e200_in12k_ft_in1k` - 87.3% top-1
  * `vit_so150m2_patch16_reg4_gap_256.sbb_e200_in12k`
* Updated InternViT-300M '2.5' weights
* Release 1.0.15

## Feb 1, 2025
* FYI PyTorch 2.6 & Python 3.13 are tested and working w/ current main and released version of `timm`

## Jan 27, 2025
* Add Kron Optimizer (PSGD w/ Kronecker-factored preconditioner) 
  * Code from https://github.com/evanatyourservice/kron_torch
  * See also https://sites.google.com/site/lixilinx/home/psgd

## Jan 19, 2025
* Fix loading of LeViT safetensor weights, remove conversion code which should have been deactivated
* Add 'SO150M' ViT weights trained with SBB recipes, decent results, but not optimal shape for ImageNet-12k/1k pretrain/ft
  * `vit_so150m_patch16_reg4_gap_256.sbb_e250_in12k_ft_in1k` - 86.7% top-1
  * `vit_so150m_patch16_reg4_gap_384.sbb_e250_in12k_ft_in1k` - 87.4% top-1
  * `vit_so150m_patch16_reg4_gap_256.sbb_e250_in12k`
* Misc typing, typo, etc. cleanup
* 1.0.14 release to get above LeViT fix out

## Jan 9, 2025
* Add support to train and validate in pure `bfloat16` or `float16`
* `wandb` project name arg added by https://github.com/caojiaolong, use arg.experiment for name
* Fix old issue w/ checkpoint saving not working on filesystem w/o hard-link support (e.g. FUSE fs mounts)
* 1.0.13 release

## Jan 6, 2025
* Add `torch.utils.checkpoint.checkpoint()` wrapper in `timm.models` that defaults `use_reentrant=False`, unless `TIMM_REENTRANT_CKPT=1` is set in env.

## Dec 31, 2024
* `convnext_nano` 384x384 ImageNet-12k pretrain & fine-tune. https://huggingface.co/models?search=convnext_nano%20r384
* Add AIM-v2 encoders from https://github.com/apple/ml-aim, see on Hub: https://huggingface.co/models?search=timm%20aimv2
* Add PaliGemma2 encoders from https://github.com/google-research/big_vision to existing PaliGemma, see on Hub: https://huggingface.co/models?search=timm%20pali2
* Add missing L/14 DFN2B 39B CLIP ViT, `vit_large_patch14_clip_224.dfn2b_s39b`
* Fix existing `RmsNorm` layer & fn to match standard formulation, use PT 2.5 impl when possible. Move old impl to `SimpleNorm` layer, it's LN w/o centering or bias. There were only two `timm` models using it, and they have been updated.
* Allow override of `cache_dir` arg for model creation
* Pass through `trust_remote_code` for HF datasets wrapper
* `inception_next_atto` model added by creator
* Adan optimizer caution, and Lamb decoupled weight decay options
* Some feature_info metadata fixed by https://github.com/brianhou0208
* All OpenCLIP and JAX (CLIP, SigLIP, Pali, etc) model weights that used load time remapping were given their own HF Hub instances so that they work with `hf-hub:` based loading, and thus will work with new Transformers `TimmWrapperModel`

## Nov 28, 2024
* More optimizers
  * Add MARS optimizer (https://arxiv.org/abs/2411.10438, https://github.com/AGI-Arena/MARS)
  * Add LaProp optimizer (https://arxiv.org/abs/2002.04839, https://github.com/Z-T-WANG/LaProp-Optimizer)
  * Add masking from 'Cautious Optimizers' (https://arxiv.org/abs/2411.16085, https://github.com/kyleliang919/C-Optim) to Adafactor, Adafactor Big Vision, AdamW (legacy), Adopt, Lamb, LaProp, Lion, NadamW, RMSPropTF, SGDW
  * Cleanup some docstrings and type annotations re optimizers and factory
* Add MobileNet-V4 Conv Medium models pretrained on in12k and fine-tuned in1k @ 384x384
  * https://huggingface.co/timm/mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k
  * https://huggingface.co/timm/mobilenetv4_conv_medium.e250_r384_in12k
  * https://huggingface.co/timm/mobilenetv4_conv_medium.e180_ad_r384_in12k
  * https://huggingface.co/timm/mobilenetv4_conv_medium.e180_r384_in12k
* Add small cs3darknet, quite good for the speed
  * https://huggingface.co/timm/cs3darknet_focus_s.ra4_e3600_r256_in1k

## Nov 12, 2024
* Optimizer factory refactor
  * New factory works by registering optimizers using an OptimInfo dataclass w/ some key traits
  * Add `list_optimizers`, `get_optimizer_class`, `get_optimizer_info` to reworked `create_optimizer_v2` fn to explore optimizers, get info or class
  * deprecate `optim.optim_factory`, move fns to `optim/_optim_factory.py` and `optim/_param_groups.py` and encourage import via `timm.optim`
* Add Adopt (https://github.com/iShohei220/adopt) optimizer
* Add 'Big Vision' variant of Adafactor (https://github.com/google-research/big_vision/blob/main/big_vision/optax.py) optimizer
* Fix original Adafactor to pick better factorization dims for convolutions
* Tweak LAMB optimizer with some improvements in torch.where functionality since original, refactor clipping a bit
* dynamic img size support in vit, deit, eva improved to support resize from non-square patch grids, thanks https://github.com/wojtke
* 
## Oct 31, 2024
Add a set of new very well trained ResNet & ResNet-V2 18/34 (basic block) weights. See https://huggingface.co/blog/rwightman/resnet-trick-or-treat

## Oct 19, 2024
* Cleanup torch amp usage to avoid cuda specific calls, merge support for Ascend (NPU) devices from [MengqingCao](https://github.com/MengqingCao) that should work now in PyTorch 2.5 w/ new device extension autoloading feature. Tested Intel Arc (XPU) in Pytorch 2.5 too and it (mostly) worked.

## Oct 16, 2024
* Fix error on importing from deprecated path `timm.models.registry`, increased priority of existing deprecation warnings to be visible
* Port weights of InternViT-300M (https://huggingface.co/OpenGVLab/InternViT-300M-448px) to `timm` as `vit_intern300m_patch14_448`

### Oct 14, 2024
* Pre-activation (ResNetV2) version of 18/18d/34/34d ResNet model defs added by request (weights pending)
* Release 1.0.10

### Oct 11, 2024
* MambaOut (https://github.com/yuweihao/MambaOut) model & weights added. A cheeky take on SSM vision models w/o the SSM (essentially ConvNeXt w/ gating). A mix of original weights + custom variations & weights.

|model                                                                                                                |img_size|top1  |top5  |param_count|
|---------------------------------------------------------------------------------------------------------------------|--------|------|------|-----------|
|[mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k](http://huggingface.co/timm/mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k)|384     |87.506|98.428|101.66     |
|[mambaout_base_plus_rw.sw_e150_in12k_ft_in1k](http://huggingface.co/timm/mambaout_base_plus_rw.sw_e150_in12k_ft_in1k)|288     |86.912|98.236|101.66     |
|[mambaout_base_plus_rw.sw_e150_in12k_ft_in1k](http://huggingface.co/timm/mambaout_base_plus_rw.sw_e150_in12k_ft_in1k)|224     |86.632|98.156|101.66     |
|[mambaout_base_tall_rw.sw_e500_in1k](http://huggingface.co/timm/mambaout_base_tall_rw.sw_e500_in1k)                  |288     |84.974|97.332|86.48      |
|[mambaout_base_wide_rw.sw_e500_in1k](http://huggingface.co/timm/mambaout_base_wide_rw.sw_e500_in1k)                  |288     |84.962|97.208|94.45      |
|[mambaout_base_short_rw.sw_e500_in1k](http://huggingface.co/timm/mambaout_base_short_rw.sw_e500_in1k)                |288     |84.832|97.27 |88.83      |
|[mambaout_base.in1k](http://huggingface.co/timm/mambaout_base.in1k)                                                  |288     |84.72 |96.93 |84.81      |
|[mambaout_small_rw.sw_e450_in1k](http://huggingface.co/timm/mambaout_small_rw.sw_e450_in1k)                          |288     |84.598|97.098|48.5       |
|[mambaout_small.in1k](http://huggingface.co/timm/mambaout_small.in1k)                                                |288     |84.5  |96.974|48.49      |
|[mambaout_base_wide_rw.sw_e500_in1k](http://huggingface.co/timm/mambaout_base_wide_rw.sw_e500_in1k)                  |224     |84.454|96.864|94.45      |
|[mambaout_base_tall_rw.sw_e500_in1k](http://huggingface.co/timm/mambaout_base_tall_rw.sw_e500_in1k)                  |224     |84.434|96.958|86.48      |
|[mambaout_base_short_rw.sw_e500_in1k](http://huggingface.co/timm/mambaout_base_short_rw.sw_e500_in1k)                |224     |84.362|96.952|88.83      |
|[mambaout_base.in1k](http://huggingface.co/timm/mambaout_base.in1k)                                                  |224     |84.168|96.68 |84.81      |
|[mambaout_small.in1k](http://huggingface.co/timm/mambaout_small.in1k)                                                |224     |84.086|96.63 |48.49      |
|[mambaout_small_rw.sw_e450_in1k](http://huggingface.co/timm/mambaout_small_rw.sw_e450_in1k)                          |224     |84.024|96.752|48.5       |
|[mambaout_tiny.in1k](http://huggingface.co/timm/mambaout_tiny.in1k)                                                  |288     |83.448|96.538|26.55      |
|[mambaout_tiny.in1k](http://huggingface.co/timm/mambaout_tiny.in1k)                                                  |224     |82.736|96.1  |26.55      |
|[mambaout_kobe.in1k](http://huggingface.co/timm/mambaout_kobe.in1k)                                                  |288     |81.054|95.718|9.14       |
|[mambaout_kobe.in1k](http://huggingface.co/timm/mambaout_kobe.in1k)                                                  |224     |79.986|94.986|9.14       |
|[mambaout_femto.in1k](http://huggingface.co/timm/mambaout_femto.in1k)                                                |288     |79.848|95.14 |7.3        |
|[mambaout_femto.in1k](http://huggingface.co/timm/mambaout_femto.in1k)                                                |224     |78.87 |94.408|7.3        |

* SigLIP SO400M ViT fine-tunes on ImageNet-1k @ 378x378, added 378x378 option for existing SigLIP 384x384 models
  *  [vit_so400m_patch14_siglip_378.webli_ft_in1k](https://huggingface.co/timm/vit_so400m_patch14_siglip_378.webli_ft_in1k) - 89.42 top-1
  *  [vit_so400m_patch14_siglip_gap_378.webli_ft_in1k](https://huggingface.co/timm/vit_so400m_patch14_siglip_gap_378.webli_ft_in1k) - 89.03
* SigLIP SO400M ViT encoder from recent multi-lingual (i18n) variant, patch16 @ 256x256 (https://huggingface.co/timm/ViT-SO400M-16-SigLIP-i18n-256). OpenCLIP update pending.
* Add two ConvNeXt 'Zepto' models & weights (one w/ overlapped stem and one w/ patch stem). Uses RMSNorm, smaller than previous 'Atto', 2.2M params.
  * [convnext_zepto_rms_ols.ra4_e3600_r224_in1k](https://huggingface.co/timm/convnext_zepto_rms_ols.ra4_e3600_r224_in1k) - 73.20 top-1 @ 224
  * [convnext_zepto_rms.ra4_e3600_r224_in1k](https://huggingface.co/timm/convnext_zepto_rms.ra4_e3600_r224_in1k) - 72.81 @ 224

### Sept 2024
* Add a suite of tiny test models for improved unit tests and niche low-resource applications (https://huggingface.co/blog/rwightman/timm-tiny-test)
* Add MobileNetV4-Conv-Small (0.5x) model (https://huggingface.co/posts/rwightman/793053396198664)
  * [mobilenetv4_conv_small_050.e3000_r224_in1k](http://hf.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k) - 65.81 top-1 @ 256, 64.76 @ 224
* Add MobileNetV3-Large variants trained with MNV4 Small recipe
  * [mobilenetv3_large_150d.ra4_e3600_r256_in1k](http://hf.co/timm/mobilenetv3_large_150d.ra4_e3600_r256_in1k) - 81.81 @ 320, 80.94 @ 256
  * [mobilenetv3_large_100.ra4_e3600_r224_in1k](http://hf.co/timm/mobilenetv3_large_100.ra4_e3600_r224_in1k) - 77.16 @ 256, 76.31 @ 224

### Aug 21, 2024
* Updated SBB ViT models trained on ImageNet-12k and fine-tuned on ImageNet-1k, challenging quite a number of much larger, slower models

| model | top1 | top5 | param_count | img_size |
| -------------------------------------------------- | ------ | ------ | ----------- | -------- |
| [vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k](https://huggingface.co/timm/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k) | 87.438 | 98.256 | 64.11 | 384 |
| [vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k](https://huggingface.co/timm/vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k) | 86.608 | 97.934 | 64.11 | 256 |
| [vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k](https://huggingface.co/timm/vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k) | 86.594 | 98.02 | 60.4 | 384 |
| [vit_betwixt_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k](https://huggingface.co/timm/vit_betwixt_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k) | 85.734 | 97.61 | 60.4 | 256 |
* MobileNet-V1 1.25, EfficientNet-B1, & ResNet50-D weights w/ MNV4 baseline challenge recipe

| model                                                                                                                    | top1   | top5   | param_count | img_size |
|--------------------------------------------------------------------------------------------------------------------------|--------|--------|-------------|----------|
| [resnet50d.ra4_e3600_r224_in1k](http://hf.co/timm/resnet50d.ra4_e3600_r224_in1k)                                         | 81.838 | 95.922 | 25.58       | 288      |
| [efficientnet_b1.ra4_e3600_r240_in1k](http://hf.co/timm/efficientnet_b1.ra4_e3600_r240_in1k)                             | 81.440 | 95.700 | 7.79        | 288      |
| [resnet50d.ra4_e3600_r224_in1k](http://hf.co/timm/resnet50d.ra4_e3600_r224_in1k)                                         | 80.952 | 95.384 | 25.58       | 224      |
| [efficientnet_b1.ra4_e3600_r240_in1k](http://hf.co/timm/efficientnet_b1.ra4_e3600_r240_in1k)                             | 80.406 | 95.152 | 7.79        | 240      |
| [mobilenetv1_125.ra4_e3600_r224_in1k](http://hf.co/timm/mobilenetv1_125.ra4_e3600_r224_in1k)                             | 77.600 | 93.804 | 6.27        | 256      |
| [mobilenetv1_125.ra4_e3600_r224_in1k](http://hf.co/timm/mobilenetv1_125.ra4_e3600_r224_in1k)                             | 76.924 | 93.234 | 6.27        | 224      |

* Add SAM2 (HieraDet) backbone arch & weight loading support
* Add Hiera Small weights trained w/ abswin pos embed on in12k & fine-tuned on 1k

|model                            |top1  |top5  |param_count|
|---------------------------------|------|------|-----------|
|hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k    |84.912|97.260|35.01      |
|hiera_small_abswin_256.sbb2_pd_e200_in12k_ft_in1k |84.560|97.106|35.01      |

### Aug 8, 2024
* Add RDNet ('DenseNets Reloaded', https://arxiv.org/abs/2403.19588), thanks [Donghyun Kim](https://github.com/dhkim0225)
  
### July 28, 2024
* Add `mobilenet_edgetpu_v2_m` weights w/ `ra4` mnv4-small based recipe. 80.1% top-1 @ 224 and 80.7 @ 256.
* Release 1.0.8

### July 26, 2024
* More MobileNet-v4 weights, ImageNet-12k pretrain w/ fine-tunes, and anti-aliased ConvLarge models

| model                                                                                            |top1  |top1_err|top5  |top5_err|param_count|img_size|
|--------------------------------------------------------------------------------------------------|------|--------|------|--------|-----------|--------|
| [mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k](http://hf.co/timm/mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k)|84.99 |15.01   |97.294|2.706   |32.59      |544     |
| [mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k](http://hf.co/timm/mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k)|84.772|15.228  |97.344|2.656   |32.59      |480     |
| [mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k](http://hf.co/timm/mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k)|84.64 |15.36   |97.114|2.886   |32.59      |448     |
| [mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k](http://hf.co/timm/mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k)|84.314|15.686  |97.102|2.898   |32.59      |384     |
| [mobilenetv4_conv_aa_large.e600_r384_in1k](http://hf.co/timm/mobilenetv4_conv_aa_large.e600_r384_in1k)     |83.824|16.176  |96.734|3.266   |32.59      |480     |
| [mobilenetv4_conv_aa_large.e600_r384_in1k](http://hf.co/timm/mobilenetv4_conv_aa_large.e600_r384_in1k)             |83.244|16.756  |96.392|3.608   |32.59      |384     |
| [mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k](http://hf.co/timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k)|82.99 |17.01   |96.67 |3.33    |11.07      |320     |
| [mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k](http://hf.co/timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k)|82.364|17.636  |96.256|3.744   |11.07      |256     |

* Impressive MobileNet-V1 and EfficientNet-B0 baseline challenges (https://huggingface.co/blog/rwightman/mobilenet-baselines)
  
| model                                                                                            |top1  |top1_err|top5  |top5_err|param_count|img_size|
|--------------------------------------------------------------------------------------------------|------|--------|------|--------|-----------|--------|
| [efficientnet_b0.ra4_e3600_r224_in1k](http://hf.co/timm/efficientnet_b0.ra4_e3600_r224_in1k)                       |79.364|20.636  |94.754|5.246   |5.29       |256     |
| [efficientnet_b0.ra4_e3600_r224_in1k](http://hf.co/timm/efficientnet_b0.ra4_e3600_r224_in1k)                       |78.584|21.416  |94.338|5.662   |5.29       |224     |    
| [mobilenetv1_100h.ra4_e3600_r224_in1k](http://hf.co/timm/mobilenetv1_100h.ra4_e3600_r224_in1k)                     |76.596|23.404  |93.272|6.728   |5.28       |256     |
| [mobilenetv1_100.ra4_e3600_r224_in1k](http://hf.co/timm/mobilenetv1_100.ra4_e3600_r224_in1k)                       |76.094|23.906  |93.004|6.996   |4.23       |256     |
| [mobilenetv1_100h.ra4_e3600_r224_in1k](http://hf.co/timm/mobilenetv1_100h.ra4_e3600_r224_in1k)                     |75.662|24.338  |92.504|7.496   |5.28       |224     |
| [mobilenetv1_100.ra4_e3600_r224_in1k](http://hf.co/timm/mobilenetv1_100.ra4_e3600_r224_in1k)                       |75.382|24.618  |92.312|7.688   |4.23       |224     |

* Prototype of `set_input_size()` added to vit and swin v1/v2 models to allow changing image size, patch size, window size after model creation.
* Improved support in swin for different size handling, in addition to `set_input_size`, `always_partition` and `strict_img_size` args have been added to `__init__` to allow more flexible input size constraints
* Fix out of order indices info for intermediate 'Getter' feature wrapper, check out or range indices for same.
* Add several `tiny` < .5M param models for testing that are actually trained on ImageNet-1k

|model                       |top1  |top1_err|top5  |top5_err|param_count|img_size|crop_pct|
|----------------------------|------|--------|------|--------|-----------|--------|--------|
|test_efficientnet.r160_in1k |47.156|52.844  |71.726|28.274  |0.36       |192     |1.0     |
|test_byobnet.r160_in1k      |46.698|53.302  |71.674|28.326  |0.46       |192     |1.0     |
|test_efficientnet.r160_in1k |46.426|53.574  |70.928|29.072  |0.36       |160     |0.875   |
|test_byobnet.r160_in1k      |45.378|54.622  |70.572|29.428  |0.46       |160     |0.875   |
|test_vit.r160_in1k|42.0  |58.0    |68.664|31.336  |0.37       |192     |1.0     |
|test_vit.r160_in1k|40.822|59.178  |67.212|32.788  |0.37       |160     |0.875   |

* Fix vit reg token init, thanks [Promisery](https://github.com/Promisery)
* Other misc fixes

### June 24, 2024
* 3 more MobileNetV4 hybrid weights with different MQA weight init scheme

| model                                                                                            |top1  |top1_err|top5  |top5_err|param_count|img_size|
|--------------------------------------------------------------------------------------------------|------|--------|------|--------|-----------|--------|
| [mobilenetv4_hybrid_large.ix_e600_r384_in1k](http://hf.co/timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k) |84.356|15.644  |96.892 |3.108  |37.76      |448     |
| [mobilenetv4_hybrid_large.ix_e600_r384_in1k](http://hf.co/timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k) |83.990|16.010  |96.702 |3.298  |37.76      |384     |
| [mobilenetv4_hybrid_medium.ix_e550_r384_in1k](http://hf.co/timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k)       |83.394|16.606  |96.760|3.240   |11.07      |448     |
| [mobilenetv4_hybrid_medium.ix_e550_r384_in1k](http://hf.co/timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k)       |82.968|17.032  |96.474|3.526   |11.07      |384     |
| [mobilenetv4_hybrid_medium.ix_e550_r256_in1k](http://hf.co/timm/mobilenetv4_hybrid_medium.ix_e550_r256_in1k)       |82.492|17.508  |96.278|3.722   |11.07      |320     |
| [mobilenetv4_hybrid_medium.ix_e550_r256_in1k](http://hf.co/timm/mobilenetv4_hybrid_medium.ix_e550_r256_in1k)       |81.446|18.554  |95.704|4.296   |11.07      |256     |
* florence2 weight loading in DaViT model

### June 12, 2024
* MobileNetV4 models and initial set of `timm` trained weights added:

| model                                                                                            |top1  |top1_err|top5  |top5_err|param_count|img_size|
|--------------------------------------------------------------------------------------------------|------|--------|------|--------|-----------|--------|
| [mobilenetv4_hybrid_large.e600_r384_in1k](http://hf.co/timm/mobilenetv4_hybrid_large.e600_r384_in1k) |84.266|15.734  |96.936 |3.064  |37.76      |448     |
| [mobilenetv4_hybrid_large.e600_r384_in1k](http://hf.co/timm/mobilenetv4_hybrid_large.e600_r384_in1k) |83.800|16.200  |96.770 |3.230  |37.76      |384     |
| [mobilenetv4_conv_large.e600_r384_in1k](http://hf.co/timm/mobilenetv4_conv_large.e600_r384_in1k) |83.392|16.608  |96.622 |3.378  |32.59      |448     |
| [mobilenetv4_conv_large.e600_r384_in1k](http://hf.co/timm/mobilenetv4_conv_large.e600_r384_in1k) |82.952|17.048  |96.266 |3.734  |32.59      |384     |
| [mobilenetv4_conv_large.e500_r256_in1k](http://hf.co/timm/mobilenetv4_conv_large.e500_r256_in1k) |82.674|17.326  |96.31 |3.69    |32.59      |320     |
| [mobilenetv4_conv_large.e500_r256_in1k](http://hf.co/timm/mobilenetv4_conv_large.e500_r256_in1k)                   |81.862|18.138  |95.69 |4.31    |32.59      |256     |
| [mobilenetv4_hybrid_medium.e500_r224_in1k](http://hf.co/timm/mobilenetv4_hybrid_medium.e500_r224_in1k)             |81.276|18.724  |95.742|4.258   |11.07      |256     |
| [mobilenetv4_conv_medium.e500_r256_in1k](http://hf.co/timm/mobilenetv4_conv_medium.e500_r256_in1k)                 |80.858|19.142  |95.768|4.232   |9.72       |320     |
| [mobilenetv4_hybrid_medium.e500_r224_in1k](http://hf.co/timm/mobilenetv4_hybrid_medium.e500_r224_in1k)             |80.442|19.558  |95.38 |4.62    |11.07      |224     |
| [mobilenetv4_conv_blur_medium.e500_r224_in1k](http://hf.co/timm/mobilenetv4_conv_blur_medium.e500_r224_in1k)       |80.142|19.858  |95.298|4.702   |9.72       |256     |
| [mobilenetv4_conv_medium.e500_r256_in1k](http://hf.co/timm/mobilenetv4_conv_medium.e500_r256_in1k)                 |79.928|20.072  |95.184|4.816   |9.72       |256     |
| [mobilenetv4_conv_medium.e500_r224_in1k](http://hf.co/timm/mobilenetv4_conv_medium.e500_r224_in1k)                 |79.808|20.192  |95.186|4.814   |9.72       |256     |
| [mobilenetv4_conv_blur_medium.e500_r224_in1k](http://hf.co/timm/mobilenetv4_conv_blur_medium.e500_r224_in1k)       |79.438|20.562  |94.932|5.068   |9.72       |224     |
| [mobilenetv4_conv_medium.e500_r224_in1k](http://hf.co/timm/mobilenetv4_conv_medium.e500_r224_in1k)                 |79.094|20.906  |94.77 |5.23    |9.72       |224     |
| [mobilenetv4_conv_small.e2400_r224_in1k](http://hf.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)                 |74.616|25.384  |92.072|7.928   |3.77       |256     |
| [mobilenetv4_conv_small.e1200_r224_in1k](http://hf.co/timm/mobilenetv4_conv_small.e1200_r224_in1k)                 |74.292|25.708  |92.116|7.884   |3.77       |256     |
| [mobilenetv4_conv_small.e2400_r224_in1k](http://hf.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)                 |73.756|26.244  |91.422|8.578   |3.77       |224     |
| [mobilenetv4_conv_small.e1200_r224_in1k](http://hf.co/timm/mobilenetv4_conv_small.e1200_r224_in1k)                 |73.454|26.546  |91.34 |8.66    |3.77       |224     |

* Apple MobileCLIP (https://arxiv.org/pdf/2311.17049, FastViT and ViT-B) image tower model support & weights added (part of OpenCLIP support).
* ViTamin (https://arxiv.org/abs/2404.02132) CLIP image tower model & weights added (part of OpenCLIP support).
* OpenAI CLIP Modified ResNet image tower modelling & weight support (via ByobNet). Refactor AttentionPool2d.

### May 14, 2024
* Support loading PaliGemma jax weights into SigLIP ViT models with average pooling.
* Add Hiera models from Meta (https://github.com/facebookresearch/hiera).
* Add `normalize=` flag for transforms, return non-normalized torch.Tensor with original dtype (for `chug`)
* Version 1.0.3 release

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
* Above feature support achieved through a new `forward_intermediates()` API that can be used with a feature wrapping module or directly.
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

## Introduction

Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arxiv papers, etc in the README, documentation, and code docstrings. Please let me know if I missed anything.

## Features

### Models

All model architecture families include variants with pretrained weights. There are specific model variants without any weights, it is NOT a bug. Help training new or better weights is always appreciated.

* Aggregating Nested Transformers - https://arxiv.org/abs/2105.12723
* BEiT - https://arxiv.org/abs/2106.08254
* BEiT-V2 - https://arxiv.org/abs/2208.06366
* BEiT3 - https://arxiv.org/abs/2208.10442
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
* EfficientFormer-V2 - https://arxiv.org/abs/2212.08059
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
* FasterNet - https://arxiv.org/abs/2303.03667
* FastViT - https://arxiv.org/abs/2303.14189
* FlexiViT - https://arxiv.org/abs/2212.08013
* FocalNet (Focal Modulation Networks) - https://arxiv.org/abs/2203.11926
* GCViT (Global Context Vision Transformer) - https://arxiv.org/abs/2206.09959
* GhostNet - https://arxiv.org/abs/1911.11907
* GhostNet-V2 - https://arxiv.org/abs/2211.12905
* GhostNet-V3 - https://arxiv.org/abs/2404.11202
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
* MambaOut - https://arxiv.org/abs/2405.07992
* MaxViT (Multi-Axis Vision Transformer) - https://arxiv.org/abs/2204.01697
* MetaFormer (PoolFormer-v2, ConvFormer, CAFormer) - https://arxiv.org/abs/2210.13452
* MLP-Mixer - https://arxiv.org/abs/2105.01601
* MobileCLIP - https://arxiv.org/abs/2311.17049
* MobileNet-V3 (MBConvNet w/ Efficient Head) - https://arxiv.org/abs/1905.02244
  * FBNet-V3 - https://arxiv.org/abs/2006.02049
  * HardCoRe-NAS - https://arxiv.org/abs/2102.11646
  * LCNet - https://arxiv.org/abs/2109.15099
* MobileNetV4 - https://arxiv.org/abs/2404.10518
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
* RDNet (DenseNets Reloaded) - https://arxiv.org/abs/2403.19588
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
* ROPE-ViT - https://arxiv.org/abs/2403.13298
* SelecSLS - https://arxiv.org/abs/1907.00837
* Selective Kernel Networks - https://arxiv.org/abs/1903.06586
* Sequencer2D - https://arxiv.org/abs/2205.01972
* SHViT - https://arxiv.org/abs/2401.16456
* SigLIP (image encoder) - https://arxiv.org/abs/2303.15343
* SigLIP 2 (image encoder) - https://arxiv.org/abs/2502.14786
* StarNet - https://arxiv.org/abs/2403.19967
* SwiftFormer - https://arxiv.org/pdf/2303.15446
* Swin S3 (AutoFormerV2) - https://arxiv.org/abs/2111.14725
* Swin Transformer - https://arxiv.org/abs/2103.14030
* Swin Transformer V2 - https://arxiv.org/abs/2111.09883
* TinyViT - https://arxiv.org/abs/2207.10666
* Transformer-iN-Transformer (TNT) - https://arxiv.org/abs/2103.00112
* TResNet - https://arxiv.org/abs/2003.13630
* Twins (Spatial Attention in Vision Transformers) - https://arxiv.org/pdf/2104.13840.pdf
* VGG - https://arxiv.org/abs/1409.1556
* Visformer - https://arxiv.org/abs/2104.12533
* Vision Transformer - https://arxiv.org/abs/2010.11929
* ViTamin - https://arxiv.org/abs/2404.02132
* VOLO (Vision Outlooker) - https://arxiv.org/abs/2106.13112
* VovNet V2 and V1 - https://arxiv.org/abs/1911.06667
* Xception - https://arxiv.org/abs/1610.02357
* Xception (Modified Aligned, Gluon) - https://arxiv.org/abs/1802.02611
* Xception (Modified Aligned, TF) - https://arxiv.org/abs/1802.02611
* XCiT (Cross-Covariance Image Transformers) - https://arxiv.org/abs/2106.09681

### Optimizers
To see full list of optimizers w/ descriptions: `timm.optim.list_optimizers(with_description=True)`

Included optimizers available via `timm.optim.create_optimizer_v2` factory method:
* `adabelief` an implementation of AdaBelief adapted from https://github.com/juntang-zhuang/Adabelief-Optimizer - https://arxiv.org/abs/2010.07468
* `adafactor` adapted from [FAIRSeq impl](https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py) - https://arxiv.org/abs/1804.04235
* `adafactorbv` adapted from [Big Vision](https://github.com/google-research/big_vision/blob/main/big_vision/optax.py) - https://arxiv.org/abs/2106.04560
* `adahessian` by [David Samuel](https://github.com/davda54/ada-hessian) - https://arxiv.org/abs/2006.00719
* `adamp` and `sgdp` by [Naver ClovAI](https://github.com/clovaai) - https://arxiv.org/abs/2006.08217
* `adan` an implementation of Adan adapted from https://github.com/sail-sg/Adan - https://arxiv.org/abs/2208.06677
* `adopt` ADOPT adapted from https://github.com/iShohei220/adopt - https://arxiv.org/abs/2411.02853
* `kron` PSGD w/ Kronecker-factored preconditioner from https://github.com/evanatyourservice/kron_torch - https://sites.google.com/site/lixilinx/home/psgd
* `lamb` an implementation of Lamb and LambC (w/ trust-clipping) cleaned up and modified to support use with XLA - https://arxiv.org/abs/1904.00962
* `laprop` optimizer from https://github.com/Z-T-WANG/LaProp-Optimizer - https://arxiv.org/abs/2002.04839
* `lars` an implementation of LARS and LARC (w/ trust-clipping) - https://arxiv.org/abs/1708.03888
* `lion` and implementation of Lion adapted from https://github.com/google/automl/tree/master/lion - https://arxiv.org/abs/2302.06675
* `lookahead` adapted from impl by [Liam](https://github.com/alphadl/lookahead.pytorch) - https://arxiv.org/abs/1907.08610
* `madgrad` an implementation of MADGRAD adapted from https://github.com/facebookresearch/madgrad - https://arxiv.org/abs/2101.11075
* `mars` MARS optimizer from https://github.com/AGI-Arena/MARS - https://arxiv.org/abs/2411.10438
* `nadam` an implementation of Adam w/ Nesterov momentum
* `nadamw` an implementation of AdamW (Adam w/ decoupled weight-decay) w/ Nesterov momentum. A simplified impl based on https://github.com/mlcommons/algorithmic-efficiency
* `novograd` by [Masashi Kimura](https://github.com/convergence-lab/novograd) - https://arxiv.org/abs/1905.11286
* `radam` by [Liyuan Liu](https://github.com/LiyuanLucasLiu/RAdam) - https://arxiv.org/abs/1908.03265
* `rmsprop_tf` adapted from PyTorch RMSProp by myself. Reproduces much improved Tensorflow RMSProp behaviour
* `sgdw` and implementation of SGD w/ decoupled weight-decay
* `fused<name>` optimizers by name with [NVIDIA Apex](https://github.com/NVIDIA/apex/tree/master/apex/optimizers) installed
* `bnb<name>` optimizers by name with [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) installed
* `cadamw`, `clion`, and more 'Cautious' optimizers from https://github.com/kyleliang919/C-Optim - https://arxiv.org/abs/2411.16085
* `adam`, `adamw`, `rmsprop`, `adadelta`, `adagrad`, and `sgd` pass through to `torch.optim` implementations
* `c` suffix (eg `adamc`, `nadamc` to implement 'corrected weight decay' in https://arxiv.org/abs/2506.02285)
  
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

[Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055-2/) by [Chris Hughes](https://github.com/Chris-hughes10) is an extensive blog post covering many aspects of `timm` in detail.

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
* lightly_train - https://github.com/lightly-ai/lightly-train

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
