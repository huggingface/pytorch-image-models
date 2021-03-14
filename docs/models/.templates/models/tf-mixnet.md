# (Tensorflow) MixNet

**MixNet** is a type of convolutional neural network discovered via AutoML that utilises [MixConvs](https://paperswithcode.com/method/mixconv) instead of regular [depthwise convolutions](https://paperswithcode.com/method/depthwise-convolution).

The weights from this model were ported from [Tensorflow/TPU](https://github.com/tensorflow/tpu).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{tan2019mixconv,
      title={MixConv: Mixed Depthwise Convolutional Kernels}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2019},
      eprint={1907.09595},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: TF MixNet
  Paper:
    Title: 'MixConv: Mixed Depthwise Convolutional Kernels'
    URL: https://paperswithcode.com/paper/mixnet-mixed-depthwise-convolutional-kernels
Models:
- Name: tf_mixnet_l
  In Collection: TF MixNet
  Metadata:
    FLOPs: 688674516
    Parameters: 7330000
    File Size: 29620756
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - MNAS
    Training Data:
    - ImageNet
    ID: tf_mixnet_l
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1720
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.78%
      Top 5 Accuracy: 94.0%
- Name: tf_mixnet_m
  In Collection: TF MixNet
  Metadata:
    FLOPs: 416633502
    Parameters: 5010000
    File Size: 20310871
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - MNAS
    Training Data:
    - ImageNet
    ID: tf_mixnet_m
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1709
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.96%
      Top 5 Accuracy: 93.16%
- Name: tf_mixnet_s
  In Collection: TF MixNet
  Metadata:
    FLOPs: 302587678
    Parameters: 4130000
    File Size: 16738218
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - MNAS
    Training Data:
    - ImageNet
    ID: tf_mixnet_s
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1698
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.68%
      Top 5 Accuracy: 92.64%
-->
