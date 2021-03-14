# SSL ResNeXT

A **ResNeXt** repeats a [building block](https://paperswithcode.com/method/resnext-block) that aggregates a set of transformations with the same topology. Compared to a [ResNet](https://paperswithcode.com/method/resnet), it exposes a new dimension,  *cardinality* (the size of the set of transformations) $C$, as an essential factor in addition to the dimensions of depth and width. 

The model in this collection utilises semi-supervised learning to improve the performance of the model. The approach brings important gains to standard architectures for image, video and fine-grained classification. 

Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/abs-1905-00546,
  author    = {I. Zeki Yalniz and
               Herv{\'{e}} J{\'{e}}gou and
               Kan Chen and
               Manohar Paluri and
               Dhruv Mahajan},
  title     = {Billion-scale semi-supervised learning for image classification},
  journal   = {CoRR},
  volume    = {abs/1905.00546},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.00546},
  archivePrefix = {arXiv},
  eprint    = {1905.00546},
  timestamp = {Mon, 28 Sep 2020 08:19:37 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1905-00546.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: SSL ResNext
  Paper:
    Title: Billion-scale semi-supervised learning for image classification
    URL: https://paperswithcode.com/paper/billion-scale-semi-supervised-learning-for
Models:
- Name: ssl_resnext101_32x16d
  In Collection: SSL ResNext
  Metadata:
    FLOPs: 46623691776
    Parameters: 194030000
    File Size: 777518664
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
    - YFCC-100M
    Training Resources: 64x GPUs
    ID: ssl_resnext101_32x16d
    LR: 0.0015
    Epochs: 30
    Layers: 101
    Crop Pct: '0.875'
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L944
  Weights: https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.84%
      Top 5 Accuracy: 96.09%
- Name: ssl_resnext101_32x4d
  In Collection: SSL ResNext
  Metadata:
    FLOPs: 10298145792
    Parameters: 44180000
    File Size: 177341913
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
    - YFCC-100M
    Training Resources: 64x GPUs
    ID: ssl_resnext101_32x4d
    LR: 0.0015
    Epochs: 30
    Layers: 101
    Crop Pct: '0.875'
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L924
  Weights: https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.91%
      Top 5 Accuracy: 95.73%
- Name: ssl_resnext101_32x8d
  In Collection: SSL ResNext
  Metadata:
    FLOPs: 21180417024
    Parameters: 88790000
    File Size: 356056638
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
    - YFCC-100M
    Training Resources: 64x GPUs
    ID: ssl_resnext101_32x8d
    LR: 0.0015
    Epochs: 30
    Layers: 101
    Crop Pct: '0.875'
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L934
  Weights: https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.61%
      Top 5 Accuracy: 96.04%
- Name: ssl_resnext50_32x4d
  In Collection: SSL ResNext
  Metadata:
    FLOPs: 5472648192
    Parameters: 25030000
    File Size: 100428550
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
    - YFCC-100M
    Training Resources: 64x GPUs
    ID: ssl_resnext50_32x4d
    LR: 0.0015
    Epochs: 30
    Layers: 50
    Crop Pct: '0.875'
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L914
  Weights: https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.3%
      Top 5 Accuracy: 95.41%
-->
