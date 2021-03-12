# Summary

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a *compound coefficient*. Unlike conventional practice that arbitrary scales  these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2^N$ times more computational resources, then we can simply increase the network depth by $\alpha ^ N$,  width by $\beta ^ N$, and image size by $\gamma ^ N$, where $\alpha, \beta, \gamma$ are constant coefficients determined by a small grid search on the original small model. EfficientNet uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a  principled way.

The compound scaling method is justified by the intuition that if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2), in addition to [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block).

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
- Name: tf_efficientnet_b1
  Metadata:
    FLOPs: 883633200
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 31512534
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b1
    LR: 0.256
    Crop Pct: '0.882'
    Momentum: 0.9
    Image Size: '240'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1251
  In Collection: TF EfficientNet
- Name: tf_efficientnet_b4
  Metadata:
    FLOPs: 5749638672
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Resources: TPUv3 Cloud TPU
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 77989689
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b4
    LR: 0.256
    Crop Pct: '0.922'
    Momentum: 0.9
    Image Size: '380'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1281
  Config: ''
  In Collection: TF EfficientNet
- Name: tf_efficientnet_b2
  Metadata:
    FLOPs: 1234321170
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 36797929
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b2
    LR: 0.256
    Crop Pct: '0.89'
    Momentum: 0.9
    Image Size: '260'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1261
  In Collection: TF EfficientNet
- Name: tf_efficientnet_b3
  Metadata:
    FLOPs: 2275247568
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 49381362
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b3
    LR: 0.256
    Crop Pct: '0.904'
    Momentum: 0.9
    Image Size: '300'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1271
  In Collection: TF EfficientNet
- Name: tf_efficientnet_b0
  Metadata:
    FLOPs: 488688572
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Resources: TPUv3 Cloud TPU
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 21383997
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b0
    LR: 0.256
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1241
  Config: ''
  In Collection: TF EfficientNet
- Name: tf_efficientnet_b5
  Metadata:
    FLOPs: 13176501888
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 122403150
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b5
    LR: 0.256
    Crop Pct: '0.934'
    Momentum: 0.9
    Image Size: '456'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1291
  In Collection: TF EfficientNet
- Name: tf_efficientnet_b6
  Metadata:
    FLOPs: 24180518488
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 173232007
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b6
    LR: 0.256
    Crop Pct: '0.942'
    Momentum: 0.9
    Image Size: '528'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1301
  In Collection: TF EfficientNet
- Name: tf_efficientnet_b7
  Metadata:
    FLOPs: 48205304880
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 266850607
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b7
    LR: 0.256
    Crop Pct: '0.949'
    Momentum: 0.9
    Image Size: '600'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1312
  In Collection: TF EfficientNet
- Name: tf_efficientnet_b8
  Metadata:
    FLOPs: 80962956270
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 351379853
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b8
    LR: 0.256
    Crop Pct: '0.954'
    Momentum: 0.9
    Image Size: '672'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1323
  In Collection: TF EfficientNet
- Name: tf_efficientnet_el
  Metadata:
    FLOPs: 9356616096
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
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 42800271
    Tasks:
    - Image Classification
    ID: tf_efficientnet_el
    Crop Pct: '0.904'
    Image Size: '300'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1551
  In Collection: TF EfficientNet
- Name: tf_efficientnet_em
  Metadata:
    FLOPs: 3636607040
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
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 27933644
    Tasks:
    - Image Classification
    ID: tf_efficientnet_em
    Crop Pct: '0.882'
    Image Size: '240'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1541
  In Collection: TF EfficientNet
- Name: tf_efficientnet_es
  Metadata:
    FLOPs: 2057577472
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
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 22008479
    Tasks:
    - Image Classification
    ID: tf_efficientnet_es
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1531
  In Collection: TF EfficientNet
- Name: tf_efficientnet_l2_ns_475
  Metadata:
    FLOPs: 217795669644
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: TPUv3 Cloud TPU
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 1925950424
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_l2_ns_475
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.936'
    Momentum: 0.9
    Image Size: '475'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1509
  Config: ''
  In Collection: TF EfficientNet
Collections:
- Name: TF EfficientNet
  Paper:
    title: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
    url: https://paperswithcode.com//paper/efficientnet-rethinking-model-scaling-for
  type: model-index
Type: model-index
-->
