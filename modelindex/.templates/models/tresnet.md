# Summary

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
Models:
- Name: tresnet_l
  Metadata:
    FLOPs: 10873416792
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
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
    File Size: 224440219
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_l
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L267
  Config: ''
  In Collection: TResNet
- Name: tresnet_l_448
  Metadata:
    FLOPs: 43488238584
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
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
    File Size: 224440219
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_l_448
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L285
  Config: ''
  In Collection: TResNet
- Name: tresnet_m
  Metadata:
    FLOPs: 5733048064
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
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
    File Size: 125861314
    Tasks:
    - Image Classification
    Training Time: < 24 hours
    ID: tresnet_m
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L261
  Config: ''
  In Collection: TResNet
- Name: tresnet_m_448
  Metadata:
    FLOPs: 22929743104
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
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
    File Size: 125861314
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_m_448
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L279
  Config: ''
  In Collection: TResNet
- Name: tresnet_xl
  Metadata:
    FLOPs: 15162534034
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
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
    File Size: 314378965
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_xl
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L273
  Config: ''
  In Collection: TResNet
- Name: tresnet_xl_448
  Metadata:
    FLOPs: 60641712730
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
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
    File Size: 224440219
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_xl_448
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L291
  Config: ''
  In Collection: TResNet
Collections:
- Name: TResNet
  Paper:
    title: 'TResNet: High Performance GPU-Dedicated Architecture'
    url: https://paperswithcode.com//paper/tresnet-high-performance-gpu-dedicated
  type: model-index
Type: model-index
-->
