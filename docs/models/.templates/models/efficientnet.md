# EfficientNet

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
Type: model-index
Collections:
- Name: EfficientNet
  Paper:
    Title: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
    URL: https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for
Models:
- Name: efficientnet_b0
  In Collection: EfficientNet
  Metadata:
    FLOPs: 511241564
    Parameters: 5290000
    File Size: 21376743
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b0
    Layers: 18
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1002
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.71%
      Top 5 Accuracy: 93.52%
- Name: efficientnet_b1
  In Collection: EfficientNet
  Metadata:
    FLOPs: 909691920
    Parameters: 7790000
    File Size: 31502706
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b1
    Crop Pct: '0.875'
    Image Size: '240'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1011
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.71%
      Top 5 Accuracy: 94.15%
- Name: efficientnet_b2
  In Collection: EfficientNet
  Metadata:
    FLOPs: 1265324514
    Parameters: 9110000
    File Size: 36788104
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b2
    Crop Pct: '0.875'
    Image Size: '260'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1020
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.38%
      Top 5 Accuracy: 95.08%
- Name: efficientnet_b2a
  In Collection: EfficientNet
  Metadata:
    FLOPs: 1452041554
    Parameters: 9110000
    File Size: 49369973
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b2a
    Crop Pct: '1.0'
    Image Size: '288'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1029
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.61%
      Top 5 Accuracy: 95.32%
- Name: efficientnet_b3
  In Collection: EfficientNet
  Metadata:
    FLOPs: 2327905920
    Parameters: 12230000
    File Size: 49369973
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b3
    Crop Pct: '0.904'
    Image Size: '300'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1038
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.08%
      Top 5 Accuracy: 96.03%
- Name: efficientnet_b3a
  In Collection: EfficientNet
  Metadata:
    FLOPs: 2600628304
    Parameters: 12230000
    File Size: 49369973
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b3a
    Crop Pct: '1.0'
    Image Size: '320'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1047
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.25%
      Top 5 Accuracy: 96.11%
- Name: efficientnet_em
  In Collection: EfficientNet
  Metadata:
    FLOPs: 3935516480
    Parameters: 6900000
    File Size: 27927309
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_em
    Crop Pct: '0.882'
    Image Size: '240'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1118
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_em_ra2-66250f76.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.26%
      Top 5 Accuracy: 94.79%
- Name: efficientnet_es
  In Collection: EfficientNet
  Metadata:
    FLOPs: 2317181824
    Parameters: 5440000
    File Size: 22003339
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_es
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1110
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_ra-f111e99c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.09%
      Top 5 Accuracy: 93.93%
- Name: efficientnet_lite0
  In Collection: EfficientNet
  Metadata:
    FLOPs: 510605024
    Parameters: 4650000
    File Size: 18820005
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_lite0
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1163
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.5%
      Top 5 Accuracy: 92.51%
-->
