# (Legacy) SE-ResNet

**SE ResNet** is a variant of a [ResNet](https://www.paperswithcode.com/method/resnet) that employs [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) to enable the network to perform dynamic channel-wise feature recalibration.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{hu2019squeezeandexcitation,
      title={Squeeze-and-Excitation Networks}, 
      author={Jie Hu and Li Shen and Samuel Albanie and Gang Sun and Enhua Wu},
      year={2019},
      eprint={1709.01507},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: Legacy SE ResNet
  Paper:
    Title: Squeeze-and-Excitation Networks
    URL: https://paperswithcode.com/paper/squeeze-and-excitation-networks
Models:
- Name: legacy_seresnet101
  In Collection: Legacy SE ResNet
  Metadata:
    FLOPs: 9762614000
    Parameters: 49330000
    File Size: 197822624
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: legacy_seresnet101
    LR: 0.6
    Epochs: 100
    Layers: 101
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L426
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet101-7e38fcc6.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.38%
      Top 5 Accuracy: 94.26%
- Name: legacy_seresnet152
  In Collection: Legacy SE ResNet
  Metadata:
    FLOPs: 14553578160
    Parameters: 66819999
    File Size: 268033864
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: legacy_seresnet152
    LR: 0.6
    Epochs: 100
    Layers: 152
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L433
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet152-d17c99b7.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.67%
      Top 5 Accuracy: 94.38%
- Name: legacy_seresnet18
  In Collection: Legacy SE ResNet
  Metadata:
    FLOPs: 2328876024
    Parameters: 11780000
    File Size: 47175663
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: legacy_seresnet18
    LR: 0.6
    Epochs: 100
    Layers: 18
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L405
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet18-4bb0ce65.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 71.74%
      Top 5 Accuracy: 90.34%
- Name: legacy_seresnet34
  In Collection: Legacy SE ResNet
  Metadata:
    FLOPs: 4706201004
    Parameters: 21960000
    File Size: 87958697
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: legacy_seresnet34
    LR: 0.6
    Epochs: 100
    Layers: 34
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L412
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet34-a4004e63.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.79%
      Top 5 Accuracy: 92.13%
- Name: legacy_seresnet50
  In Collection: Legacy SE ResNet
  Metadata:
    FLOPs: 4974351024
    Parameters: 28090000
    File Size: 112611220
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: legacy_seresnet50
    LR: 0.6
    Epochs: 100
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bilinear
    Minibatch Size: 1024
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L419
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet50-ce0d4300.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.64%
      Top 5 Accuracy: 93.74%
-->
