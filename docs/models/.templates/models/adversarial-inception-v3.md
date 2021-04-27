# Adversarial Inception v3

**Inception v3** is a convolutional neural network architecture from the Inception family that makes several improvements including using [Label Smoothing](https://paperswithcode.com/method/label-smoothing), Factorized 7 x 7 convolutions, and the use of an [auxiliary classifer](https://paperswithcode.com/method/auxiliary-classifier) to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead). The key building block is an [Inception Module](https://paperswithcode.com/method/inception-v3-module).

This particular model was trained for study of adversarial examples (adversarial training).

The weights from this model were ported from [Tensorflow/Models](https://github.com/tensorflow/models).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/abs-1804-00097,
  author    = {Alexey Kurakin and
               Ian J. Goodfellow and
               Samy Bengio and
               Yinpeng Dong and
               Fangzhou Liao and
               Ming Liang and
               Tianyu Pang and
               Jun Zhu and
               Xiaolin Hu and
               Cihang Xie and
               Jianyu Wang and
               Zhishuai Zhang and
               Zhou Ren and
               Alan L. Yuille and
               Sangxia Huang and
               Yao Zhao and
               Yuzhe Zhao and
               Zhonglin Han and
               Junjiajia Long and
               Yerkebulan Berdibekov and
               Takuya Akiba and
               Seiya Tokui and
               Motoki Abe},
  title     = {Adversarial Attacks and Defences Competition},
  journal   = {CoRR},
  volume    = {abs/1804.00097},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.00097},
  archivePrefix = {arXiv},
  eprint    = {1804.00097},
  timestamp = {Thu, 31 Oct 2019 16:31:22 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1804-00097.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: Adversarial Inception v3
  Paper:
    Title: Adversarial Attacks and Defences Competition
    URL: https://paperswithcode.com/paper/adversarial-attacks-and-defences-competition
Models:
- Name: adv_inception_v3
  In Collection: Adversarial Inception v3
  Metadata:
    FLOPs: 7352418880
    Parameters: 23830000
    File Size: 95549439
    Architecture:
    - 1x1 Convolution
    - Auxiliary Classifier
    - Average Pooling
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inception-v3 Module
    - Max Pooling
    - ReLU
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: adv_inception_v3
    Crop Pct: '0.875'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/inception_v3.py#L456
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/adv_inception_v3-9e27bd63.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.58%
      Top 5 Accuracy: 93.74%
-->
