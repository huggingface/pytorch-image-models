# ResNeXt

A **ResNeXt** repeats a [building block](https://paperswithcode.com/method/resnext-block) that aggregates a set of transformations with the same topology. Compared to a [ResNet](https://paperswithcode.com/method/resnet), it exposes a new dimension,  *cardinality* (the size of the set of transformations) $C$, as an essential factor in addition to the dimensions of depth and width. 

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
- Name: ResNeXt
  Paper:
    Title: Aggregated Residual Transformations for Deep Neural Networks
    URL: https://paperswithcode.com/paper/aggregated-residual-transformations-for-deep
Models:
- Name: resnext101_32x8d
  In Collection: ResNeXt
  Metadata:
    FLOPs: 21180417024
    Parameters: 88790000
    File Size: 356082095
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
    ID: resnext101_32x8d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L877
  Weights: https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.3%
      Top 5 Accuracy: 94.53%
- Name: resnext50_32x4d
  In Collection: ResNeXt
  Metadata:
    FLOPs: 5472648192
    Parameters: 25030000
    File Size: 100435887
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
    ID: resnext50_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L851
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d_ra-d733960d.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.79%
      Top 5 Accuracy: 94.61%
- Name: resnext50d_32x4d
  In Collection: ResNeXt
  Metadata:
    FLOPs: 5781119488
    Parameters: 25050000
    File Size: 100515304
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
    ID: resnext50d_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L869
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.67%
      Top 5 Accuracy: 94.87%
- Name: tv_resnext50_32x4d
  In Collection: ResNeXt
  Metadata:
    FLOPs: 5472648192
    Parameters: 25030000
    File Size: 100441675
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
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tv_resnext50_32x4d
    LR: 0.1
    Epochs: 90
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Batch Size: 32
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L842
  Weights: https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.61%
      Top 5 Accuracy: 93.68%
-->
