# Summary

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
Models:
- Name: hrnet_w18_small
  Metadata:
    FLOPs: 2071651488
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 52934302
    Tasks:
    - Image Classification
    Training Time: ''
    ID: hrnet_w18_small
    Layers: 18
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L790
  Config: ''
  In Collection: HRNet
- Name: hrnet_w18_small_v2
  Metadata:
    FLOPs: 3360023160
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 62682879
    Tasks:
    - Image Classification
    Training Time: ''
    ID: hrnet_w18_small_v2
    Layers: 18
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L795
  Config: ''
  In Collection: HRNet
- Name: hrnet_w32
  Metadata:
    FLOPs: 11524528320
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 165547812
    Tasks:
    - Image Classification
    Training Time: 60 hours
    ID: hrnet_w32
    Layers: 32
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L810
  Config: ''
  In Collection: HRNet
- Name: hrnet_w40
  Metadata:
    FLOPs: 16381182192
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 230899236
    Tasks:
    - Image Classification
    Training Time: ''
    ID: hrnet_w40
    Layers: 40
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L815
  Config: ''
  In Collection: HRNet
- Name: hrnet_w44
  Metadata:
    FLOPs: 19202520264
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 268957432
    Tasks:
    - Image Classification
    Training Time: ''
    ID: hrnet_w44
    Layers: 44
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L820
  Config: ''
  In Collection: HRNet
- Name: hrnet_w48
  Metadata:
    FLOPs: 22285865760
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 310603710
    Tasks:
    - Image Classification
    Training Time: 80 hours
    ID: hrnet_w48
    Layers: 48
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L825
  Config: ''
  In Collection: HRNet
- Name: hrnet_w18
  Metadata:
    FLOPs: 5547205500
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 85718883
    Tasks:
    - Image Classification
    Training Time: ''
    ID: hrnet_w18
    Layers: 18
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L800
  Config: ''
  In Collection: HRNet
- Name: hrnet_w64
  Metadata:
    FLOPs: 37239321984
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 513071818
    Tasks:
    - Image Classification
    Training Time: ''
    ID: hrnet_w64
    Layers: 64
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L830
  Config: ''
  In Collection: HRNet
- Name: hrnet_w30
  Metadata:
    FLOPs: 10474119492
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - ReLU
    - Residual Connection
    File Size: 151452218
    Tasks:
    - Image Classification
    Training Time: ''
    ID: hrnet_w30
    Layers: 30
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/hrnet.py#L805
  Config: ''
  In Collection: HRNet
Collections:
- Name: HRNet
  Paper:
    title: Deep High-Resolution Representation Learning for Visual Recognition
    url: https://papperswithcode.com//paper/190807919
  type: model-index
Type: model-index
-->
