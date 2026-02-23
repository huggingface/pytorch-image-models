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

## Feb 23, 2026
* Add token distillation training support to distillation task wrappers
* Remove some torch.jit usage in prep for official deprecation
* Caution added to AdamP optimizer
* Call reset_parameters() even if meta-device init so that buffers get init w/ hacks like init_empty_weights
* Tweak Muon optimizer to work with DTensor/FSDP2 (clamp_ instead of clamp_min_, alternate NS branch for DTensor)
* Release 1.0.25

## Jan 21, 2026
* **Compat Break**: Fix oversight w/ QKV vs MLP bias in `ParallelScalingBlock` (& `DiffParallelScalingBlock`)
  * Does not impact any trained `timm` models but could impact downstream use.

## Jan 5 & 6, 2026
* Release 1.0.24
* Add new benchmark result csv files for inference timing on all models w/ RTX Pro 6000, 5090, and 4090 cards w/ PyTorch 2.9.1
* Fix moved module error in deprecated timm.models.layers import path that impacts legacy imports
* Release 1.0.23

## Dec 30, 2025
* Add better NAdaMuon trained `dpwee`, `dwee`, `dlittle` (differential) ViTs with a small boost over previous runs
  * https://huggingface.co/timm/vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k (83.24% top-1)
  * https://huggingface.co/timm/vit_dwee_patch16_reg1_gap_256.sbb_nadamuon_in1k  (81.80% top-1)
  * https://huggingface.co/timm/vit_dpwee_patch16_reg1_gap_256.sbb_nadamuon_in1k (81.67% top-1)
* Add a ~21M param `timm` variant of the CSATv2 model at 512x512 & 640x640
  * https://huggingface.co/timm/csatv2_21m.sw_r640_in1k (83.13% top-1)
  * https://huggingface.co/timm/csatv2_21m.sw_r512_in1k (82.58% top-1)
* Factor non-persistent param init out of `__init__` into a common method that can be externally called via `init_non_persistent_buffers()` after meta-device init. 
  
## Dec 12, 2025
* Add CSATV2 model (thanks https://github.com/gusdlf93) -- a lightweight but high res model with DCT stem & spatial attention. https://huggingface.co/Hyunil/CSATv2
* Add AdaMuon and NAdaMuon optimizer support to existing `timm` Muon impl. Appears more competitive vs AdamW with familiar hparams for image tasks.
* End of year PR cleanup, merge aspects of several long open PR
  * Merge differential attention (`DiffAttention`), add corresponding `DiffParallelScalingBlock` (for ViT), train some wee vits
    * https://huggingface.co/timm/vit_dwee_patch16_reg1_gap_256.sbb_in1k
    * https://huggingface.co/timm/vit_dpwee_patch16_reg1_gap_256.sbb_in1k
  * Add a few pooling modules, `LsePlus` and `SimPool`
  * Cleanup, optimize `DropBlock2d` (also add support to ByobNet based models)
* Bump unit tests to PyTorch 2.9.1 + Python 3.13 on upper end, lower still PyTorch 1.13 + Python 3.10
  
## Dec 1, 2025
* Add lightweight task abstraction, add logits and feature distillation support to train script via new tasks.
* Remove old APEX AMP support

## Nov 4, 2025
* Fix LayerScale / LayerScale2d init bug (init values ignored), introduced in 1.0.21. Thanks https://github.com/Ilya-Fradlin
* Release 1.0.22

## Oct 31, 2025 🎃
* Update imagenet & OOD variant result csv files to include a few new models and verify correctness over several torch & timm versions
* EfficientNet-X and EfficientNet-H B5 model weights added as part of a hparam search for AdamW vs Muon (still iterating on Muon runs)

## Oct 16-20, 2025
* Add an impl of the Muon optimizer (based on https://github.com/KellerJordan/Muon) with customizations
  * extra flexibility and improved handling for conv weights and fallbacks for weight shapes not suited for orthogonalization
  * small speedup for NS iterations by reducing allocs and using fused (b)add(b)mm ops
  * by default uses AdamW (or NAdamW if `nesterov=True`) updates if muon not suitable for parameter shape (or excluded via param group flag)
  * like torch impl, select from several LR scale adjustment fns via `adjust_lr_fn`
  * select from several NS coefficient presets or specify your own via `ns_coefficients`
* First 2 steps of 'meta' device model initialization supported
  * Fix several ops that were breaking creation under 'meta' device context
  * Add device & dtype factory kwarg support to all models and modules (anything inherting from nn.Module) in `timm`
* License fields added to pretrained cfgs in code
* Release 1.0.21

## Sept 21, 2025
* Remap DINOv3 ViT weight tags from `lvd_1689m` -> `lvd1689m` to match (same for `sat_493m` -> `sat493m`)
* Release 1.0.20

## Sept 17, 2025
* DINOv3 (https://arxiv.org/abs/2508.10104) ConvNeXt and ViT models added. ConvNeXt models were mapped to existing `timm` model. ViT support done via the EVA base model w/ a new `RotaryEmbeddingDinoV3` to match the DINOv3 specific RoPE impl
  * HuggingFace Hub: https://huggingface.co/collections/timm/timm-dinov3-68cb08bb0bee365973d52a4d
* MobileCLIP-2 (https://arxiv.org/abs/2508.20691) vision encoders. New MCI3/MCI4 FastViT variants added and weights mapped to existing FastViT and B, L/14 ViTs.
* MetaCLIP-2 Worldwide (https://arxiv.org/abs/2507.22062) ViT encoder weights added.
* SigLIP-2 (https://arxiv.org/abs/2502.14786) NaFlex ViT encoder weights added via timm NaFlexViT model.
* Misc fixes and contributions

## July 23, 2025
* Add `set_input_size()` method to EVA models, used by OpenCLIP 3.0.0 to allow resizing for timm based encoder models.
* Release 1.0.18, needed for PE-Core S & T models in OpenCLIP 3.0.0
* Fix small typing issue that broke Python 3.9 compat. 1.0.19 patch release.

## July 21, 2025
* ROPE support added to NaFlexViT. All models covered by the EVA base (`eva.py`) including EVA, EVA02, Meta PE ViT, `timm` SBB ViT w/ ROPE, and Naver ROPE-ViT can be now loaded in NaFlexViT when `use_naflex=True` passed at model creation time
* More Meta PE ViT encoders added, including small/tiny variants, lang variants w/ tiling, and more spatial variants.
* PatchDropout fixed with NaFlexViT and also w/ EVA models (regression after adding Naver ROPE-ViT)
* Fix XY order with grid_indexing='xy', impacted non-square image use in 'xy' mode (only ROPE-ViT and PE impacted).

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
* PE (Perception Encoder) - https://arxiv.org/abs/2504.13181
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
* `adamuon` and `nadamuon` as per https://github.com/Chongjie-Si/AdaMuon - https://arxiv.org/abs/2507.11005
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
* `muon` MUON optimizer from https://github.com/KellerJordan/Muon with numerous additions and improved non-transformer behaviour
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
* Space-to-Depth by [mrT23](https://github.com/mrT23/TResNet/blob/master/src/models/tresnet/layers/space_to_depth.py) (https://arxiv.org/abs/1801.04590)
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
