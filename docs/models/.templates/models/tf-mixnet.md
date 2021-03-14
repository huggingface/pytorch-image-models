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
Models:
- Name: tf_mixnet_l
  Metadata:
    FLOPs: 688674516
    Training Data:
    - ImageNet
    Training Techniques:
    - MNAS
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 29620756
    Tasks:
    - Image Classification
    ID: tf_mixnet_l
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1720
  In Collection: TF MixNet
- Name: tf_mixnet_m
  Metadata:
    FLOPs: 416633502
    Training Data:
    - ImageNet
    Training Techniques:
    - MNAS
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 20310871
    Tasks:
    - Image Classification
    ID: tf_mixnet_m
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1709
  In Collection: TF MixNet
- Name: tf_mixnet_s
  Metadata:
    FLOPs: 302587678
    Training Data:
    - ImageNet
    Training Techniques:
    - MNAS
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 16738218
    Tasks:
    - Image Classification
    ID: tf_mixnet_s
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1698
  In Collection: TF MixNet
Collections:
- Name: TF MixNet
  Paper:
    title: 'MixConv: Mixed Depthwise Convolutional Kernels'
    url: https://paperswithcode.com//paper/mixnet-mixed-depthwise-convolutional-kernels
  type: model-index
Type: model-index
-->
