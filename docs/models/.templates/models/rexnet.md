# RexNet

**Rank Expansion Networks** (ReXNets) follow a set of new design principles for designing bottlenecks in image classification models. Authors refine each layer by 1) expanding the input channel size of the convolution layer and 2) replacing the [ReLU6s](https://www.paperswithcode.com/method/relu6).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{han2020rexnet,
      title={ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network}, 
      author={Dongyoon Han and Sangdoo Yun and Byeongho Heo and YoungJoon Yoo},
      year={2020},
      eprint={2007.00992},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: RexNet
  Paper:
    Title: 'ReXNet: Diminishing Representational Bottleneck on Convolutional Neural
      Network'
    URL: https://paperswithcode.com/paper/rexnet-diminishing-representational
Models:
- Name: rexnet_100
  In Collection: RexNet
  Metadata:
    FLOPs: 509989377
    Parameters: 4800000
    File Size: 19417552
    Architecture:
    - Batch Normalization
    - Convolution
    - Dropout
    - ReLU6
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - Linear Warmup With Cosine Annealing
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: rexnet_100
    LR: 0.5
    Epochs: 400
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/rexnet.py#L212
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_100-1b4dddf4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.86%
      Top 5 Accuracy: 93.88%
- Name: rexnet_130
  In Collection: RexNet
  Metadata:
    FLOPs: 848364461
    Parameters: 7560000
    File Size: 30508197
    Architecture:
    - Batch Normalization
    - Convolution
    - Dropout
    - ReLU6
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - Linear Warmup With Cosine Annealing
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: rexnet_130
    LR: 0.5
    Epochs: 400
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/rexnet.py#L218
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_130-590d768e.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.49%
      Top 5 Accuracy: 94.67%
- Name: rexnet_150
  In Collection: RexNet
  Metadata:
    FLOPs: 1122374469
    Parameters: 9730000
    File Size: 39227315
    Architecture:
    - Batch Normalization
    - Convolution
    - Dropout
    - ReLU6
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - Linear Warmup With Cosine Annealing
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: rexnet_150
    LR: 0.5
    Epochs: 400
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/rexnet.py#L224
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_150-bd1a6aa8.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.31%
      Top 5 Accuracy: 95.16%
- Name: rexnet_200
  In Collection: RexNet
  Metadata:
    FLOPs: 1960224938
    Parameters: 16370000
    File Size: 65862221
    Architecture:
    - Batch Normalization
    - Convolution
    - Dropout
    - ReLU6
    - Residual Connection
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - Linear Warmup With Cosine Annealing
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x NVIDIA V100 GPUs
    ID: rexnet_200
    LR: 0.5
    Epochs: 400
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/rexnet.py#L230
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_200-8c0b7f2d.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.63%
      Top 5 Accuracy: 95.67%
-->
