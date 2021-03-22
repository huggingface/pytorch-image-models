# TResNet

A **TResNet** is a variant on a [ResNet](https://paperswithcode.com/method/resnet) that aim to boost accuracy while maintaining GPU training and inference efficiency.  They contain several design tricks including a SpaceToDepth stem, [Anti-Alias downsampling](https://paperswithcode.com/method/anti-alias-downsampling), In-Place Activated BatchNorm, Blocks selection and [squeeze-and-excitation layers](https://paperswithcode.com/method/squeeze-and-excitation-block).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{ridnik2020tresnet,
      title={TResNet: High Performance GPU-Dedicated Architecture}, 
      author={Tal Ridnik and Hussam Lawen and Asaf Noy and Emanuel Ben Baruch and Gilad Sharir and Itamar Friedman},
      year={2020},
      eprint={2003.13630},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: TResNet
  Paper:
    Title: 'TResNet: High Performance GPU-Dedicated Architecture'
    URL: https://paperswithcode.com/paper/tresnet-high-performance-gpu-dedicated
Models:
- Name: tresnet_l
  In Collection: TResNet
  Metadata:
    FLOPs: 10873416792
    Parameters: 53456696
    File Size: 224440219
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA 100 GPUs
    ID: tresnet_l
    LR: 0.01
    Epochs: 300
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L267
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_l_81_5-235b486c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.49%
      Top 5 Accuracy: 95.62%
- Name: tresnet_l_448
  In Collection: TResNet
  Metadata:
    FLOPs: 43488238584
    Parameters: 53456696
    File Size: 224440219
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA 100 GPUs
    ID: tresnet_l_448
    LR: 0.01
    Epochs: 300
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L285
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_l_448-940d0cd1.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.26%
      Top 5 Accuracy: 95.98%
- Name: tresnet_m
  In Collection: TResNet
  Metadata:
    FLOPs: 5733048064
    Parameters: 41282200
    File Size: 125861314
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA 100 GPUs
    Training Time: < 24 hours
    ID: tresnet_m
    LR: 0.01
    Epochs: 300
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L261
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_m_80_8-dbc13962.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.8%
      Top 5 Accuracy: 94.86%
- Name: tresnet_m_448
  In Collection: TResNet
  Metadata:
    FLOPs: 22929743104
    Parameters: 29278464
    File Size: 125861314
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA 100 GPUs
    ID: tresnet_m_448
    LR: 0.01
    Epochs: 300
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L279
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_m_448-bc359d10.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.72%
      Top 5 Accuracy: 95.57%
- Name: tresnet_xl
  In Collection: TResNet
  Metadata:
    FLOPs: 15162534034
    Parameters: 75646610
    File Size: 314378965
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA 100 GPUs
    ID: tresnet_xl
    LR: 0.01
    Epochs: 300
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L273
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_xl_82_0-a2d51b00.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.05%
      Top 5 Accuracy: 95.93%
- Name: tresnet_xl_448
  In Collection: TResNet
  Metadata:
    FLOPs: 60641712730
    Parameters: 75646610
    File Size: 224440219
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA 100 GPUs
    ID: tresnet_xl_448
    LR: 0.01
    Epochs: 300
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L291
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_l_448-940d0cd1.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 83.06%
      Top 5 Accuracy: 96.19%
-->
