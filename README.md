# PyTorch Image Models, etc

## Introduction 

For each competition, personal, or freelance project involving images + Convolution Neural Networks, I build on top of an evolving collection of code and models. This repo contains a (somewhat) cleaned up and paired down iteration of that code. Hopefully it'll be of use to others.

The work of many others is present here. I've tried to make sure all source material is acknowledged:
* Training/validation scripts evolved from early versions of the [PyTorch Imagenet Examples](https://github.com/pytorch/examples)
* CUDA specific performance enhancements have been pulled from [NVIDIA's APEX Examples](https://github.com/NVIDIA/apex/tree/master/examples)
* Models are from a wide variety of sources
    * [Torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models)
    * [Cadene's Pretrained Models](https://github.com/Cadene/pretrained-models.pytorch)
    * [Myself](https://github.com/rwightman/pytorch-dpn-pretrained)
* LR scheduler ideas from [AllenNLP](https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers), [FAIRseq](https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler), and SGDR: Stochastic Gradient Descent with Warm Restarts (https://arxiv.org/abs/1608.03983)
* Random Erasing from [Zhun Zhong](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py)  (https://arxiv.org/abs/1708.04896)

## Models

I've included a few of my favourite models, but this is not an exhaustive collection. You can't do better than Cadene's collection in that regard. Most models do have pretrained weights from their respective sources or original authors. 

* ResNet/ResNeXt (from [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models) with mods by myself)
    * ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNeXt50 (32x4d), ResNeXt101 (32x4d and 64x4d)
    * 'Bag of Tricks' / Gluon C, D, E, S variations (https://arxiv.org/abs/1812.01187)
    * Instagram trained / ImageNet tuned ResNeXt101-32x8d to 32x48d from from [facebookresearch](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/)
* DenseNet (from [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models))
    * DenseNet-121, DenseNet-169, DenseNet-201, DenseNet-161
* Squeeze-and-Excitation ResNet/ResNeXt (from [Cadene](https://github.com/Cadene/pretrained-models.pytorch) with some pretrained weight additions by myself)
    * SENet-154, SE-ResNet-18, SE-ResNet-34, SE-ResNet-50, SE-ResNet-101, SE-ResNet-152, SE-ResNeXt-26 (32x4d), SE-ResNeXt50 (32x4d), SE-ResNeXt101 (32x4d)
* Inception-ResNet-V2 and Inception-V4 (from [Cadene](https://github.com/Cadene/pretrained-models.pytorch) )
* Xception (from [Cadene](https://github.com/Cadene/pretrained-models.pytorch))
* PNasNet & NASNet-A (from [Cadene](https://github.com/Cadene/pretrained-models.pytorch))
* DPN (from [me](https://github.com/rwightman/pytorch-dpn-pretrained), weights hosted by Cadene)
    * DPN-68, DPN-68b, DPN-92, DPN-98, DPN-131, DPN-107
* Generic EfficientNet (from my standalone [GenMobileNet](https://github.com/rwightman/genmobilenet-pytorch)) - A generic model that implements many of the efficient models that utilize similar DepthwiseSeparable and InvertedResidual blocks
    * EfficientNet (B0-B7) (https://arxiv.org/abs/1905.11946) -- validated, compat with TF weights
    * EfficientNet-EdgeTPU (S, M, L) (https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html) --validated w/ TF weights
    * MixNet (https://arxiv.org/abs/1907.09595) -- validated, compat with TF weights
    * MNASNet B1, A1 (Squeeze-Excite), and Small (https://arxiv.org/abs/1807.11626)
    * MobileNet-V1 (https://arxiv.org/abs/1704.04861)
    * MobileNet-V2 (https://arxiv.org/abs/1801.04381)
    * MobileNet-V3 (https://arxiv.org/abs/1905.02244) -- pretrained model good, still no official impl to verify against
    * ChamNet (https://arxiv.org/abs/1812.08934) -- specific arch details hard to find, currently an educated guess
    * FBNet-C (https://arxiv.org/abs/1812.03443) -- TODO A/B variants
    * Single-Path NAS (https://arxiv.org/abs/1904.02877) -- pixel1 variant

Use the  `--model` arg to specify model for train, validation, inference scripts. Match the all lowercase
creation fn for the model you'd like.

## Features
Several (less common) features that I often utilize in my projects are included. Many of their additions are the reason why I maintain my own set of models, instead of using others' via PIP:
* All models have a common default configuration interface and API for
    * accessing/changing the classifier - `get_classifier` and `reset_classifier`
    * doing a forward pass on just the features - `forward_features`
    * these makes it easy to write consistent network wrappers that work with any of the models
* All models have a consistent pretrained weight loader that adapts last linear if necessary, and from 3 to 1 channel input if desired
* The train script works in several process/GPU modes:
    * NVIDIA DDP w/ a single GPU per process, multiple processes with APEX present (AMP mixed-precision optional)
    * PyTorch DistributedDataParallel w/ multi-gpu, single process (AMP disabled as it crashes when enabled)
    * PyTorch w/ single GPU single process (AMP optional)
* A dynamic global pool implementation that allows selecting from average pooling, max pooling, average + max, or concat([average, max]) at model creation. All global pooling is adaptive average by default and compatible with pretrained weights.
* A 'Test Time Pool' wrapper that can wrap any of the included models and usually provide improved performance doing inference with input images larger than the training size. Idea adapted from original DPN implementation when I ported (https://github.com/cypw/DPNs)
* Training schedules and techniques that provide competitive results (Cosine LR, Random Erasing, Label Smoothing, etc)
* Mixup (as in https://arxiv.org/abs/1710.09412) - currently implementing/testing
* An inference script that dumps output to CSV is provided as an example

## Results

A CSV file containing an ImageNet-1K validation results summary for all included models with pretrained weights and default configurations is located [here](results/results-all.csv)

### Self-trained Weights
I've leveraged the training scripts in this repository to train a few of the models with missing weights to good levels of performance. These numbers are all for 224x224 training and validation image sizing with the usual 87.5% validation crop.

|Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling  | Image Size |
|---|---|---|---|---|---|
| mixnet_xl | 80.120 (19.880) | 95.022 (4.978) | 11.90M | bicubic | 224 |
| efficientnet_b2 | 79.760 (20.240) | 94.714 (5.286) | 9.11M | bicubic | 260 |
| resnext50d_32x4d | 79.674 (20.326) | 94.868 (5.132) | 25.1M | bicubic | 224 |
| mixnet_l | 78.976 (21.024 | 94.184 (5.816) | 7.33M | bicubic | 224 |
| efficientnet_b1 | 78.692 (21.308) | 94.086 (5.914) | 7.79M | bicubic | 240 |
| resnext50_32x4d | 78.512 (21.488) | 94.042 (5.958) | 25M | bicubic | 224 |
| resnet50 | 78.470 (21.530) | 94.266 (5.734) | 25.6M | bicubic | 224 |
| mixnet_m | 77.256 (22.744) | 93.418 (6.582) | 5.01M | bicubic | 224 |
| seresnext26_32x4d | 77.104 (22.896) | 93.316 (6.684) | 16.8M | bicubic | 224 |
| efficientnet_b0 | 76.912 (23.088) | 93.210 (6.790) | 5.29M | bicubic | 224 |
| resnet26d | 76.68 (23.32) | 93.166 (6.834) | 16M | bicubic | 224 |
| mixnet_s | 75.988 (24.012) | 92.794 (7.206) | 4.13M | bicubic | 224 |
| mobilenetv3_100 | 75.634 (24.366) | 92.708 (7.292) | 5.5M | bicubic | 224 |
| mnasnet_a1 | 75.448 (24.552) | 92.604 (7.396) | 3.89M | bicubic | 224 |
| resnet26 | 75.292 (24.708) | 92.57 (7.43) | 16M | bicubic | 224 |
| fbnetc_100 | 75.124 (24.876) | 92.386 (7.614) | 5.6M | bilinear | 224 |
| resnet34 | 75.110 (24.890) | 92.284 (7.716) | 22M | bilinear | 224 |
| seresnet34 | 74.808 (25.192) | 92.124 (7.876) | 22M | bilinear | 224 |
| mnasnet_b1 | 74.658 (25.342) | 92.114 (7.886) | 4.38M | bicubic | 224 |
| spnasnet_100 | 74.084 (25.916)  | 91.818 (8.182) | 4.42M | bilinear | 224 |
| seresnet18 | 71.742 (28.258) | 90.334 (9.666) | 11.8M | bicubic | 224 |

### Ported Weights

| Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling | Image Size | Source |
|---|---|---|---|---|---|---|
| tf_efficientnet_b7 *tfp  | 84.480 (15.520) | 96.870 (3.130) | 66.35  | bicubic | 600 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b7       | 84.420 (15.580) | 96.906 (3.094) | 66.35  | bicubic | 600 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b6 *tfp  | 84.140 (15.860) | 96.852 (3.148) | 43.04  | bicubic | 528 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b6       | 84.110 (15.890) | 96.886 (3.114) | 43.04  | bicubic | 528 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b5 *tfp  | 83.694 (16.306) | 96.696 (3.304) | 30.39  | bicubic | 456 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b5       | 83.688 (16.312) | 96.714 (3.286) | 30.39  | bicubic | 456 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b4       | 83.022 (16.978) | 96.300 (3.700) | 19.34  | bicubic | 380 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b4 *tfp  | 82.948 (17.052) | 96.308 (3.692) | 19.34  | bicubic | 380 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b3 *tfp  | 81.576 (18.424) | 95.662 (4.338) | 12.23  | bicubic | 300 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b3       | 81.636 (18.364) | 95.718 (4.282) | 12.23  | bicubic | 300 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| gluon_senet154           | 81.224 (18.776) | 95.356 (4.644) | 115.09 | bicubic | 224 | |
| gluon_resnet152_v1s      | 81.012 (18.988) | 95.416 (4.584) | 60.32  | bicubic | 224 | |
| gluon_seresnext101_32x4d | 80.902 (19.098) | 95.294 (4.706) | 48.96  | bicubic | 224 | |
| gluon_seresnext101_64x4d | 80.890 (19.110) | 95.304 (4.696) | 88.23  | bicubic | 224 | |
| gluon_resnext101_64x4d   | 80.602 (19.398) | 94.994 (5.006) | 83.46  | bicubic | 224 | |
| tf_efficientnet_el       | 80.534 (19.466) | 95.190 (4.810) | 10.59 | bicubic | 300 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu) |
| tf_efficientnet_el *tfp  | 80.476 (19.524) | 95.200 (4.800) | 10.59 | bicubic | 300 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu) |
| gluon_resnet152_v1d      | 80.470 (19.530) | 95.206 (4.794) | 60.21  | bicubic | 224 | |
| gluon_resnet101_v1d      | 80.424 (19.576) | 95.020 (4.980) | 44.57  | bicubic | 224 | |
| gluon_resnext101_32x4d   | 80.334 (19.666) | 94.926 (5.074) | 44.18  | bicubic | 224 | |
| gluon_resnet101_v1s      | 80.300 (19.700) | 95.150 (4.850) | 44.67  | bicubic | 224 | |
| tf_efficientnet_b2 *tfp  | 80.188 (19.812) | 94.974 (5.026) | 9.11  | bicubic | 260 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_efficientnet_b2       | 80.086 (19.914) | 94.908 (5.092) | 9.11  | bicubic | 260 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| gluon_resnet152_v1c      | 79.916 (20.084) | 94.842 (5.158) | 60.21  | bicubic | 224 | |
| gluon_seresnext50_32x4d  | 79.912 (20.088) | 94.818 (5.182) | 27.56  | bicubic | 224 | |
| gluon_resnet152_v1b      | 79.692 (20.308) | 94.738 (5.262) | 60.19  | bicubic | 224 | |
| gluon_xception65         | 79.604 (20.396) | 94.748 (5.252) | 39.92  | bicubic | 299 | |
| gluon_resnet101_v1c      | 79.544 (20.456) | 94.586 (5.414) | 44.57  | bicubic | 224 | |
| gluon_resnext50_32x4d    | 79.356 (20.644) | 94.424 (5.576) | 25.03  | bicubic | 224 | |
| gluon_resnet101_v1b      | 79.304 (20.696) | 94.524 (5.476) | 44.55  | bicubic | 224 | |
| tf_efficientnet_b1 *tfp  | 79.172 (20.828) | 94.450 (5.550) | 7.79  | bicubic | 240 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| gluon_resnet50_v1d       | 79.074 (20.926) | 94.476 (5.524) | 25.58  | bicubic | 224 | |
| tf_efficientnet_em *tfp  | 78.958 (21.042) | 94.458 (5.542) | 6.90 | bicubic | 240 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu) |
| tf_mixnet_l *tfp         | 78.846 (21.154) | 94.212 (5.788) | 7.33  | bilinear | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) |
| tf_efficientnet_b1       | 78.826 (21.174) | 94.198 (5.802) | 7.79  | bicubic | 240 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| gluon_inception_v3       | 78.804 (21.196) | 94.380 (5.620) | 27.16M | bicubic | 299 | [MxNet Gluon](https://gluon-cv.mxnet.io/model_zoo/classification.html) |
| tf_mixnet_l              | 78.770 (21.230) | 94.004 (5.996) | 7.33  | bicubic | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) |
| tf_efficientnet_em       | 78.742 (21.258) | 94.332 (5.668) | 6.90 | bicubic | 240 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu) |
| gluon_resnet50_v1s       | 78.712 (21.288) | 94.242 (5.758) | 25.68  | bicubic | 224 | |
| gluon_resnet50_v1c       | 78.010 (21.990) | 93.988 (6.012) | 25.58  | bicubic | 224 | |
| tf_inception_v3          | 77.856 (22.144) | 93.644 (6.356) | 27.16M | bicubic | 299 | [Tensorflow Slim](https://github.com/tensorflow/models/tree/master/research/slim) |
| tf_efficientnet_es *tfp  | 77.616 (22.384) | 93.750 (6.250) | 5.44 | bicubic | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu) |
| gluon_resnet50_v1b       | 77.578 (22.422) | 93.718 (6.282) | 25.56  | bicubic | 224 | |
| adv_inception_v3         | 77.576 (22.424) | 93.724 (6.276) | 27.16M | bicubic | 299 | [Tensorflow Adv models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models) |
| tf_efficientnet_es       | 77.264 (22.736) | 93.600 (6.400) | 5.44 | bicubic | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu) |
| tf_efficientnet_b0 *tfp  | 77.258 (22.742) | 93.478 (6.522) | 5.29  | bicubic | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_mixnet_m *tfp         | 77.072 (22.928) | 93.368 (6.632) | 5.01  | bilinear | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) |
| tf_mixnet_m              | 76.950 (23.050) | 93.156 (6.844) | 5.01  | bicubic | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) |
| tf_efficientnet_b0       | 76.848 (23.152) | 93.228 (6.772) | 5.29  | bicubic | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| tf_mixnet_s *tfp         | 75.800 (24.200) | 92.788 (7.212) | 4.13  | bilinear | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) |
| tf_mixnet_s              | 75.648 (24.352) | 92.636 (7.364) | 4.13  | bicubic | 224 | [Google](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) |
| gluon_resnet34_v1b       | 74.580 (25.420) | 91.988 (8.012) | 21.80 | bicubic | 224 | |
| gluon_resnet18_v1b       | 70.830 (29.170) | 89.756 (10.244) | 11.69 | bicubic | 224 | |

Models with `*tfp` next to them were scored with `--tf-preprocessing` flag. 

The `tf_efficientnet`, `tf_mixnet` models require an equivalent for 'SAME' padding as their arch results in asymmetric padding. I've added this in the model creation wrapper, but it does come with a performance penalty. 

## Usage

### Environment

All development and testing has been done in Conda Python 3 environments on Linux x86-64 systems, specifically Python 3.6.x and 3.7.x. Little to no care has been taken to be Python 2.x friendly and I don't plan to support it. If you run into any challenges running on Windows, or other OS, I'm definitely open to looking into those issues so long as it's in a reproducible (read Conda) environment.

PyTorch versions 1.0 and 1.1 have been tested with this code. 

I've tried to keep the dependencies minimal, the setup is as per the PyTorch default install instructions for Conda:
```
conda create -n torch-env
conda activate torch-env
conda install -c pytorch pytorch torchvision cudatoolkit=10.0
```

### Pip
This package can be installed via pip. Currently, the model factory (`timm.create_model`) is the most useful component to use via a pip install.

Install (after conda env/install):
```
pip install timm
```

Use:
```
>>> import timm
>>> m = timm.create_model('mobilenetv3_100', pretrained=True)
>>> m.eval()
```

### Scripts
A train, validation, inference, and checkpoint cleaning script included in the github root folder. Scripts are not currently packaged in the pip release.

#### Training

The variety of training args is large and not all combinations of options (or even options) have been fully tested. For the training dataset folder, specify the folder to the base that contains a `train` and `validation` folder.

To train an SE-ResNet34 on ImageNet, locally distributed, 4 GPUs, one process per GPU w/ cosine schedule, random-erasing prob of 50% and per-pixel random value:

`./distributed_train.sh 4 /data/imagenet --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 -j 4`

NOTE: NVIDIA APEX should be installed to run in per-process distributed via DDP or to enable AMP mixed precision with the --amp flag
 
#### Validation / Inference

Validation and inference scripts are similar in usage. One outputs metrics on a validation set and the other outputs topk class ids in a csv. Specify the folder containing validation images, not the base as in training script. 

To validate with the model's pretrained weights (if they exist):

`python validate.py /imagenet/validation/ --model seresnext26_32x4d --pretrained`

To run inference from a checkpoint:

`python inference.py /imagenet/validation/ --model mobilenetv3_100 --checkpoint ./output/model_best.pth.tar`

## TODO
A number of additions planned in the future for various projects, incl
* Do a model performance (speed + accuracy) benchmarking across all models (make runable as script)
* Add usage examples to comments, good hyper params for training
* Comments, cleanup and the usual things that get pushed back
