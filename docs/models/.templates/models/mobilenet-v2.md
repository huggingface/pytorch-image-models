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
Type: model-index
Collections:
- Name: MobileNet V2
  Paper:
    Title: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks'
    URL: https://paperswithcode.com/paper/mobilenetv2-inverted-residuals-and-linear
Models:
- Name: mobilenetv2_100
  In Collection: MobileNet V2
  Metadata:
    FLOPs: 401920448
    Parameters: 3500000
    File Size: 14202571
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
    Tasks:
    - Image Classification
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 16x GPUs
    ID: mobilenetv2_100
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L955
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 72.95%
      Top 5 Accuracy: 91.0%
- Name: mobilenetv2_110d
  In Collection: MobileNet V2
  Metadata:
    FLOPs: 573958832
    Parameters: 4520000
    File Size: 18316431
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
    Tasks:
    - Image Classification
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 16x GPUs
    ID: mobilenetv2_110d
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L969
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_110d_ra-77090ade.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.05%
      Top 5 Accuracy: 92.19%
- Name: mobilenetv2_120d
  In Collection: MobileNet V2
  Metadata:
    FLOPs: 888510048
    Parameters: 5830000
    File Size: 23651121
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
    Tasks:
    - Image Classification
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 16x GPUs
    ID: mobilenetv2_120d
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L977
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_120d_ra-5987e2ed.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.28%
      Top 5 Accuracy: 93.51%
- Name: mobilenetv2_140
  In Collection: MobileNet V2
  Metadata:
    FLOPs: 770196784
    Parameters: 6110000
    File Size: 24673555
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
    Tasks:
    - Image Classification
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 16x GPUs
    ID: mobilenetv2_140
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L962
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_140_ra-21a4e913.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.51%
      Top 5 Accuracy: 93.0%
-->
