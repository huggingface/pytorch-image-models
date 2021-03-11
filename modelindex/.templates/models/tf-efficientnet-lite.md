# Summary

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a *compound coefficient*. Unlike conventional practice that arbitrary scales  these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2^N$ times more computational resources, then we can simply increase the network depth by $\alpha ^ N$,  width by $\beta ^ N$, and image size by $\gamma ^ N$, where $\alpha, \beta, \gamma$ are constant coefficients determined by a small grid search on the original small model. EfficientNet uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a  principled way.

The compound scaling method is justified by the intuition that if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2).

EfficientNet-Lite makes EfficientNet more suitable for mobile devices by introducing [ReLU6](https://paperswithcode.com/method/relu6) activation functions and removing [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation).

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
Models:
- Name: tf_efficientnet_lite3
  Metadata:
    FLOPs: 2011534304
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    File Size: 33161413
    Tasks:
    - Image Classification
    ID: tf_efficientnet_lite3
    Crop Pct: '0.904'
    Image Size: '300'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1629
  In Collection: TF EfficientNet Lite
- Name: tf_efficientnet_lite4
  Metadata:
    FLOPs: 5164802912
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    File Size: 52558819
    Tasks:
    - Image Classification
    ID: tf_efficientnet_lite4
    Crop Pct: '0.92'
    Image Size: '380'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1640
  In Collection: TF EfficientNet Lite
- Name: tf_efficientnet_lite2
  Metadata:
    FLOPs: 1068494432
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    File Size: 24658687
    Tasks:
    - Image Classification
    ID: tf_efficientnet_lite2
    Crop Pct: '0.89'
    Image Size: '260'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1618
  In Collection: TF EfficientNet Lite
- Name: tf_efficientnet_lite1
  Metadata:
    FLOPs: 773639520
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    File Size: 21939331
    Tasks:
    - Image Classification
    ID: tf_efficientnet_lite1
    Crop Pct: '0.882'
    Image Size: '240'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1607
  In Collection: TF EfficientNet Lite
- Name: tf_efficientnet_lite0
  Metadata:
    FLOPs: 488052032
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    File Size: 18820223
    Tasks:
    - Image Classification
    ID: tf_efficientnet_lite0
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1596
  In Collection: TF EfficientNet Lite
Collections:
- Name: TF EfficientNet Lite
  Paper:
    title: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
    url: https://papperswithcode.com//paper/efficientnet-rethinking-model-scaling-for
  type: model-index
Type: model-index
-->
