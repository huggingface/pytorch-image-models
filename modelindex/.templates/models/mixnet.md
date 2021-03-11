# Summary

**MixNet** is a type of convolutional neural network discovered via AutoML that utilises [MixConvs](https://paperswithcode.com/method/mixconv) instead of regular [depthwise convolutions](https://paperswithcode.com/method/depthwise-convolution).

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
- Name: mixnet_xl
  Metadata:
    FLOPs: 1195880424
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
    File Size: 48001170
    Tasks:
    - Image Classification
    ID: mixnet_xl
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1678
  In Collection: MixNet
- Name: mixnet_m
  Metadata:
    FLOPs: 454543374
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
    File Size: 20298347
    Tasks:
    - Image Classification
    ID: mixnet_m
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1660
  In Collection: MixNet
- Name: mixnet_s
  Metadata:
    FLOPs: 321264910
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
    File Size: 16727982
    Tasks:
    - Image Classification
    ID: mixnet_s
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1651
  In Collection: MixNet
- Name: mixnet_l
  Metadata:
    FLOPs: 738671316
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
    File Size: 29608232
    Tasks:
    - Image Classification
    ID: mixnet_l
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1669
  In Collection: MixNet
Collections:
- Name: MixNet
  Paper:
    title: 'MixConv: Mixed Depthwise Convolutional Kernels'
    url: https://papperswithcode.com//paper/mixnet-mixed-depthwise-convolutional-kernels
  type: model-index
Type: model-index
-->
