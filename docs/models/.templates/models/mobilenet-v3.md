# MobileNet v3

**MobileNetV3** is a convolutional neural network that is designed for mobile phone CPUs. The network design includes the use of a [hard swish activation](https://paperswithcode.com/method/hard-swish) and [squeeze-and-excitation](https://paperswithcode.com/method/squeeze-and-excitation-block) modules in the [MBConv blocks](https://paperswithcode.com/method/inverted-residual-block).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/abs-1905-02244,
  author    = {Andrew Howard and
               Mark Sandler and
               Grace Chu and
               Liang{-}Chieh Chen and
               Bo Chen and
               Mingxing Tan and
               Weijun Wang and
               Yukun Zhu and
               Ruoming Pang and
               Vijay Vasudevan and
               Quoc V. Le and
               Hartwig Adam},
  title     = {Searching for MobileNetV3},
  journal   = {CoRR},
  volume    = {abs/1905.02244},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.02244},
  archivePrefix = {arXiv},
  eprint    = {1905.02244},
  timestamp = {Tue, 12 Jan 2021 15:30:06 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1905-02244.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: MobileNet V3
  Paper:
    Title: Searching for MobileNetV3
    URL: https://paperswithcode.com/paper/searching-for-mobilenetv3
Models:
- Name: mobilenetv3_large_100
  In Collection: MobileNet V3
  Metadata:
    FLOPs: 287193752
    Parameters: 5480000
    File Size: 22076443
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Hard Swish
    - Inverted Residual Block
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x4 TPU Pod
    ID: mobilenetv3_large_100
    LR: 0.1
    Dropout: 0.8
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L363
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.77%
      Top 5 Accuracy: 92.54%
- Name: mobilenetv3_rw
  In Collection: MobileNet V3
  Metadata:
    FLOPs: 287190638
    Parameters: 5480000
    File Size: 22064048
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Hard Swish
    - Inverted Residual Block
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x4 TPU Pod
    ID: mobilenetv3_rw
    LR: 0.1
    Dropout: 0.8
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L384
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.62%
      Top 5 Accuracy: 92.71%
-->
