# MixNet

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
Type: model-index
Collections:
- Name: MixNet
  Paper:
    Title: 'MixConv: Mixed Depthwise Convolutional Kernels'
    URL: https://paperswithcode.com/paper/mixnet-mixed-depthwise-convolutional-kernels
Models:
- Name: mixnet_l
  In Collection: MixNet
  Metadata:
    FLOPs: 738671316
    Parameters: 7330000
    File Size: 29608232
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
    ID: mixnet_l
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1669
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.98%
      Top 5 Accuracy: 94.18%
- Name: mixnet_m
  In Collection: MixNet
  Metadata:
    FLOPs: 454543374
    Parameters: 5010000
    File Size: 20298347
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
    ID: mixnet_m
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1660
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.27%
      Top 5 Accuracy: 93.42%
- Name: mixnet_s
  In Collection: MixNet
  Metadata:
    FLOPs: 321264910
    Parameters: 4130000
    File Size: 16727982
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
    ID: mixnet_s
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1651
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.99%
      Top 5 Accuracy: 92.79%
- Name: mixnet_xl
  In Collection: MixNet
  Metadata:
    FLOPs: 1195880424
    Parameters: 11900000
    File Size: 48001170
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
    ID: mixnet_xl
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1678
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.47%
      Top 5 Accuracy: 94.93%
-->
