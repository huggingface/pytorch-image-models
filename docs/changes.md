# Recent Changes

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
