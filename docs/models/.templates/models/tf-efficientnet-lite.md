# (Tensorflow) EfficientNet Lite

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a *compound coefficient*. Unlike conventional practice that arbitrary scales  these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2^N$ times more computational resources, then we can simply increase the network depth by $\alpha ^ N$,  width by $\beta ^ N$, and image size by $\gamma ^ N$, where $\alpha, \beta, \gamma$ are constant coefficients determined by a small grid search on the original small model. EfficientNet uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a  principled way.

The compound scaling method is justified by the intuition that if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2).

EfficientNet-Lite makes EfficientNet more suitable for mobile devices by introducing [ReLU6](https://paperswithcode.com/method/relu6) activation functions and removing [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation).

The weights from this model were ported from [Tensorflow/TPU](https://github.com/tensorflow/tpu).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{tan2020efficientnet,
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2020},
      eprint={1905.11946},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!--
Type: model-index
Collections:
- Name: TF EfficientNet Lite
  Paper:
    Title: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
    URL: https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for
Models:
- Name: tf_efficientnet_lite0
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 488052032
    Parameters: 4650000
    File Size: 18820223
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite0
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1596
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.83%
      Top 5 Accuracy: 92.17%
- Name: tf_efficientnet_lite1
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 773639520
    Parameters: 5420000
    File Size: 21939331
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite1
    Crop Pct: '0.882'
    Image Size: '240'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1607
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.67%
      Top 5 Accuracy: 93.24%
- Name: tf_efficientnet_lite2
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 1068494432
    Parameters: 6090000
    File Size: 24658687
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite2
    Crop Pct: '0.89'
    Image Size: '260'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1618
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite2-dcccb7df.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.48%
      Top 5 Accuracy: 93.75%
- Name: tf_efficientnet_lite3
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 2011534304
    Parameters: 8199999
    File Size: 33161413
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite3
    Crop Pct: '0.904'
    Image Size: '300'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1629
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.83%
      Top 5 Accuracy: 94.91%
- Name: tf_efficientnet_lite4
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 5164802912
    Parameters: 13010000
    File Size: 52558819
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite4
    Crop Pct: '0.92'
    Image Size: '380'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1640
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite4-741542c3.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.54%
      Top 5 Accuracy: 95.66%
-->
