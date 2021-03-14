# MobileNet v2

**MobileNetV2** is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an [inverted residual structure](https://paperswithcode.com/method/inverted-residual-block) where the residual connections are between the bottleneck layers.  The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/abs-1801-04381,
  author    = {Mark Sandler and
               Andrew G. Howard and
               Menglong Zhu and
               Andrey Zhmoginov and
               Liang{-}Chieh Chen},
  title     = {Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification,
               Detection and Segmentation},
  journal   = {CoRR},
  volume    = {abs/1801.04381},
  year      = {2018},
  url       = {http://arxiv.org/abs/1801.04381},
  archivePrefix = {arXiv},
  eprint    = {1801.04381},
  timestamp = {Tue, 12 Jan 2021 15:30:06 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1801-04381.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Models:
- Name: mobilenetv2_100
  Metadata:
    FLOPs: 401920448
    Batch Size: 1536
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 16x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - Inverted Residual Block
    - Max Pooling
    - ReLU6
    - Residual Connection
    - Softmax
    File Size: 14202571
    Tasks:
    - Image Classification
    ID: mobilenetv2_100
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L955
  In Collection: MobileNet V2
- Name: mobilenetv2_110d
  Metadata:
    FLOPs: 573958832
    Batch Size: 1536
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 16x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - Inverted Residual Block
    - Max Pooling
    - ReLU6
    - Residual Connection
    - Softmax
    File Size: 18316431
    Tasks:
    - Image Classification
    ID: mobilenetv2_110d
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L969
  In Collection: MobileNet V2
- Name: mobilenetv2_120d
  Metadata:
    FLOPs: 888510048
    Batch Size: 1536
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 16x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - Inverted Residual Block
    - Max Pooling
    - ReLU6
    - Residual Connection
    - Softmax
    File Size: 23651121
    Tasks:
    - Image Classification
    ID: mobilenetv2_120d
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L977
  In Collection: MobileNet V2
- Name: mobilenetv2_140
  Metadata:
    FLOPs: 770196784
    Batch Size: 1536
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 16x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - Inverted Residual Block
    - Max Pooling
    - ReLU6
    - Residual Connection
    - Softmax
    File Size: 24673555
    Tasks:
    - Image Classification
    ID: mobilenetv2_140
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L962
  In Collection: MobileNet V2
Collections:
- Name: MobileNet V2
  Paper:
    title: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks'
    url: https://paperswithcode.com//paper/mobilenetv2-inverted-residuals-and-linear
  type: model-index
Type: model-index
-->
