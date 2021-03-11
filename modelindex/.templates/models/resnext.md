# Summary

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
Models:
- Name: resnext101_32x8d
  Metadata:
    FLOPs: 21180417024
    Training Data:
    - ImageNet
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
    File Size: 356082095
    Tasks:
    - Image Classification
    ID: resnext101_32x8d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L877
  In Collection: ResNeXt
- Name: resnext50_32x4d
  Metadata:
    FLOPs: 5472648192
    Training Data:
    - ImageNet
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
    File Size: 100435887
    Tasks:
    - Image Classification
    ID: resnext50_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L851
  In Collection: ResNeXt
- Name: tv_resnext50_32x4d
  Metadata:
    FLOPs: 5472648192
    Epochs: 90
    Batch Size: 32
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
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
    File Size: 100441675
    Tasks:
    - Image Classification
    ID: tv_resnext50_32x4d
    LR: 0.1
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L842
  In Collection: ResNeXt
- Name: resnext50d_32x4d
  Metadata:
    FLOPs: 5781119488
    Training Data:
    - ImageNet
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
    File Size: 100515304
    Tasks:
    - Image Classification
    ID: resnext50d_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L869
  In Collection: ResNeXt
Collections:
- Name: ResNeXt
  Paper:
    title: Aggregated Residual Transformations for Deep Neural Networks
    url: https://papperswithcode.com//paper/aggregated-residual-transformations-for-deep
  type: model-index
Type: model-index
-->
