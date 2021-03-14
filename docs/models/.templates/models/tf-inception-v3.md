# (Tensorflow) Inception v3

**Inception v3** is a convolutional neural network architecture from the Inception family that makes several improvements including using [Label Smoothing](https://paperswithcode.com/method/label-smoothing), Factorized 7 x 7 convolutions, and the use of an [auxiliary classifer](https://paperswithcode.com/method/auxiliary-classifier) to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead). The key building block is an [Inception Module](https://paperswithcode.com/method/inception-v3-module).

The weights from this model were ported from [Tensorflow/Models](https://github.com/tensorflow/models).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/SzegedyVISW15,
  author    = {Christian Szegedy and
               Vincent Vanhoucke and
               Sergey Ioffe and
               Jonathon Shlens and
               Zbigniew Wojna},
  title     = {Rethinking the Inception Architecture for Computer Vision},
  journal   = {CoRR},
  volume    = {abs/1512.00567},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.00567},
  archivePrefix = {arXiv},
  eprint    = {1512.00567},
  timestamp = {Mon, 13 Aug 2018 16:49:07 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/SzegedyVISW15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: TF Inception v3
  Paper:
    Title: Rethinking the Inception Architecture for Computer Vision
    URL: https://paperswithcode.com/paper/rethinking-the-inception-architecture-for
Models:
- Name: tf_inception_v3
  In Collection: TF Inception v3
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
    Training Techniques:
    - Gradient Clipping
    - Label Smoothing
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 50x NVIDIA Kepler GPUs
    ID: tf_inception_v3
    LR: 0.045
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/inception_v3.py#L449
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_inception_v3-e0069de4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.87%
      Top 5 Accuracy: 93.65%
-->
