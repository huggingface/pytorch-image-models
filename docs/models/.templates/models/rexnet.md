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
Models:
- Name: rexnet_100
  Metadata:
    FLOPs: 509989377
    Epochs: 400
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - Linear Warmup With Cosine Annealing
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - Dropout
    - ReLU6
    - Residual Connection
    File Size: 19417552
    Tasks:
    - Image Classification
    Training Time: ''
    ID: rexnet_100
    LR: 0.5
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/rexnet.py#L212
  Config: ''
  In Collection: RexNet
- Name: rexnet_130
  Metadata:
    FLOPs: 848364461
    Epochs: 400
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - Linear Warmup With Cosine Annealing
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - Dropout
    - ReLU6
    - Residual Connection
    File Size: 30508197
    Tasks:
    - Image Classification
    Training Time: ''
    ID: rexnet_130
    LR: 0.5
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/rexnet.py#L218
  Config: ''
  In Collection: RexNet
- Name: rexnet_150
  Metadata:
    FLOPs: 1122374469
    Epochs: 400
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - Linear Warmup With Cosine Annealing
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - Dropout
    - ReLU6
    - Residual Connection
    File Size: 39227315
    Tasks:
    - Image Classification
    Training Time: ''
    ID: rexnet_150
    LR: 0.5
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/rexnet.py#L224
  Config: ''
  In Collection: RexNet
- Name: rexnet_200
  Metadata:
    FLOPs: 1960224938
    Epochs: 400
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - Linear Warmup With Cosine Annealing
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 4x NVIDIA V100 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - Dropout
    - ReLU6
    - Residual Connection
    File Size: 65862221
    Tasks:
    - Image Classification
    Training Time: ''
    ID: rexnet_200
    LR: 0.5
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/rexnet.py#L230
  Config: ''
  In Collection: RexNet
Collections:
- Name: RexNet
  Paper:
    title: 'ReXNet: Diminishing Representational Bottleneck on Convolutional Neural
      Network'
    url: https://paperswithcode.com//paper/rexnet-diminishing-representational
  type: model-index
Type: model-index
-->
