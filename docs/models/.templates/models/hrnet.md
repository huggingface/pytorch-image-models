# HRNet

**HRNet**, or **High-Resolution Net**, is a general purpose convolutional neural network for tasks like semantic segmentation, object detection and image classification. It is able to maintain high resolution representations through the whole process. We start from a high-resolution convolution stream, gradually add high-to-low resolution convolution streams one by one, and connect the multi-resolution streams in parallel. The resulting network consists of several ($4$ in the paper) stages and the $n$th stage contains $n$ streams corresponding to $n$ resolutions. The authors conduct repeated multi-resolution fusions by exchanging the information across the parallel streams over and over.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{sun2019highresolution,
      title={High-Resolution Representations for Labeling Pixels and Regions}, 
      author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
      year={2019},
      eprint={1904.04514},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: HRNet
  Paper:
    Title: Deep High-Resolution Representation Learning for Visual Recognition
    URL: https://paperswithcode.com/paper/190807919
Models:
- Name: hrnet_w18
  In Collection: HRNet
  Metadata:
    FLOPs: 5547205500
    Parameters: 21300000
    File Size: 85718883
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: hrnet_w18
    Epochs: 100
    Layers: 18
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L800
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w18-8cb57bb9.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.76%
      Top 5 Accuracy: 93.44%
- Name: hrnet_w18_small
  In Collection: HRNet
  Metadata:
    FLOPs: 2071651488
    Parameters: 13190000
    File Size: 52934302
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: hrnet_w18_small
    Epochs: 100
    Layers: 18
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L790
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnet_w18_small_v1-f460c6bc.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 72.34%
      Top 5 Accuracy: 90.68%
- Name: hrnet_w18_small_v2
  In Collection: HRNet
  Metadata:
    FLOPs: 3360023160
    Parameters: 15600000
    File Size: 62682879
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: hrnet_w18_small_v2
    Epochs: 100
    Layers: 18
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L795
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnet_w18_small_v2-4c50a8cb.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.11%
      Top 5 Accuracy: 92.41%
- Name: hrnet_w30
  In Collection: HRNet
  Metadata:
    FLOPs: 10474119492
    Parameters: 37710000
    File Size: 151452218
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: hrnet_w30
    Epochs: 100
    Layers: 30
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L805
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w30-8d7f8dab.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.21%
      Top 5 Accuracy: 94.22%
- Name: hrnet_w32
  In Collection: HRNet
  Metadata:
    FLOPs: 11524528320
    Parameters: 41230000
    File Size: 165547812
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    Training Time: 60 hours
    ID: hrnet_w32
    Epochs: 100
    Layers: 32
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L810
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w32-90d8c5fb.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.45%
      Top 5 Accuracy: 94.19%
- Name: hrnet_w40
  In Collection: HRNet
  Metadata:
    FLOPs: 16381182192
    Parameters: 57560000
    File Size: 230899236
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: hrnet_w40
    Epochs: 100
    Layers: 40
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L815
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w40-7cd397a4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.93%
      Top 5 Accuracy: 94.48%
- Name: hrnet_w44
  In Collection: HRNet
  Metadata:
    FLOPs: 19202520264
    Parameters: 67060000
    File Size: 268957432
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: hrnet_w44
    Epochs: 100
    Layers: 44
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L820
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w44-c9ac8c18.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.89%
      Top 5 Accuracy: 94.37%
- Name: hrnet_w48
  In Collection: HRNet
  Metadata:
    FLOPs: 22285865760
    Parameters: 77470000
    File Size: 310603710
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    Training Time: 80 hours
    ID: hrnet_w48
    Epochs: 100
    Layers: 48
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L825
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w48-abd2e6ab.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.32%
      Top 5 Accuracy: 94.51%
- Name: hrnet_w64
  In Collection: HRNet
  Metadata:
    FLOPs: 37239321984
    Parameters: 128060000
    File Size: 513071818
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: hrnet_w64
    Epochs: 100
    Layers: 64
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L830
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w64-b47cc881.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.46%
      Top 5 Accuracy: 94.65%
-->
