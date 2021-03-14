# (Gluon) ResNeXt

A **ResNeXt** repeats a [building block](https://paperswithcode.com/method/resnext-block) that aggregates a set of transformations with the same topology. Compared to a [ResNet](https://paperswithcode.com/method/resnet), it exposes a new dimension,  *cardinality* (the size of the set of transformations) $C$, as an essential factor in addition to the dimensions of depth and width. 

The weights from this model were ported from [Gluon](https://cv.gluon.ai/model_zoo/classification.html).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/XieGDTH16,
  author    = {Saining Xie and
               Ross B. Girshick and
               Piotr Doll{\'{a}}r and
               Zhuowen Tu and
               Kaiming He},
  title     = {Aggregated Residual Transformations for Deep Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1611.05431},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.05431},
  archivePrefix = {arXiv},
  eprint    = {1611.05431},
  timestamp = {Mon, 13 Aug 2018 16:45:58 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/XieGDTH16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: Gloun ResNeXt
  Paper:
    Title: Aggregated Residual Transformations for Deep Neural Networks
    URL: https://paperswithcode.com/paper/aggregated-residual-transformations-for-deep
Models:
- Name: gluon_resnext101_32x4d
  In Collection: Gloun ResNeXt
  Metadata:
    FLOPs: 10298145792
    Parameters: 44180000
    File Size: 177367414
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnext101_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L193
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_32x4d-b253c8c4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.33%
      Top 5 Accuracy: 94.91%
- Name: gluon_resnext101_64x4d
  In Collection: Gloun ResNeXt
  Metadata:
    FLOPs: 19954172928
    Parameters: 83460000
    File Size: 334737852
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnext101_64x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L201
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_64x4d-f9a8e184.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.63%
      Top 5 Accuracy: 95.0%
- Name: gluon_resnext50_32x4d
  In Collection: Gloun ResNeXt
  Metadata:
    FLOPs: 5472648192
    Parameters: 25030000
    File Size: 100441719
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnext50_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L185
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext50_32x4d-e6a097c1.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.35%
      Top 5 Accuracy: 94.42%
-->
