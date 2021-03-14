# (Tensorflow) MobileNet v3

**MobileNetV3** is a convolutional neural network that is designed for mobile phone CPUs. The network design includes the use of a [hard swish activation](https://paperswithcode.com/method/hard-swish) and [squeeze-and-excitation](https://paperswithcode.com/method/squeeze-and-excitation-block) modules in the [MBConv blocks](https://paperswithcode.com/method/inverted-residual-block).

The weights from this model were ported from [Tensorflow/Models](https://github.com/tensorflow/models).

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
- Name: TF MobileNet V3
  Paper:
    Title: Searching for MobileNetV3
    URL: https://paperswithcode.com/paper/searching-for-mobilenetv3
Models:
- Name: tf_mobilenetv3_large_075
  In Collection: TF MobileNet V3
  Metadata:
    FLOPs: 194323712
    Parameters: 3990000
    File Size: 16097377
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
    ID: tf_mobilenetv3_large_075
    LR: 0.1
    Dropout: 0.8
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L394
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 73.45%
      Top 5 Accuracy: 91.34%
- Name: tf_mobilenetv3_large_100
  In Collection: TF MobileNet V3
  Metadata:
    FLOPs: 274535288
    Parameters: 5480000
    File Size: 22076649
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
    ID: tf_mobilenetv3_large_100
    LR: 0.1
    Dropout: 0.8
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L403
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.51%
      Top 5 Accuracy: 92.61%
- Name: tf_mobilenetv3_large_minimal_100
  In Collection: TF MobileNet V3
  Metadata:
    FLOPs: 267216928
    Parameters: 3920000
    File Size: 15836368
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
    ID: tf_mobilenetv3_large_minimal_100
    LR: 0.1
    Dropout: 0.8
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L412
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 72.24%
      Top 5 Accuracy: 90.64%
- Name: tf_mobilenetv3_small_075
  In Collection: TF MobileNet V3
  Metadata:
    FLOPs: 48457664
    Parameters: 2040000
    File Size: 8242701
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
    Training Resources: 16x GPUs
    ID: tf_mobilenetv3_small_075
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bilinear
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L421
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 65.72%
      Top 5 Accuracy: 86.13%
- Name: tf_mobilenetv3_small_100
  In Collection: TF MobileNet V3
  Metadata:
    FLOPs: 65450600
    Parameters: 2540000
    File Size: 10256398
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
    Training Resources: 16x GPUs
    ID: tf_mobilenetv3_small_100
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bilinear
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L430
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 67.92%
      Top 5 Accuracy: 87.68%
- Name: tf_mobilenetv3_small_minimal_100
  In Collection: TF MobileNet V3
  Metadata:
    FLOPs: 60827936
    Parameters: 2040000
    File Size: 8258083
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
    Training Resources: 16x GPUs
    ID: tf_mobilenetv3_small_minimal_100
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bilinear
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L439
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 62.91%
      Top 5 Accuracy: 84.24%
-->
