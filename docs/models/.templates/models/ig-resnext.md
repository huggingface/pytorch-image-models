# Instagram ResNeXt WSL

A **ResNeXt** repeats a [building block](https://paperswithcode.com/method/resnext-block) that aggregates a set of transformations with the same topology. Compared to a [ResNet](https://paperswithcode.com/method/resnet), it exposes a new dimension,  *cardinality* (the size of the set of transformations) $C$, as an essential factor in addition to the dimensions of depth and width. 

This model was trained on billions of Instagram images using thousands of distinct hashtags as labels exhibit excellent transfer learning performance. 

Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{mahajan2018exploring,
      title={Exploring the Limits of Weakly Supervised Pretraining}, 
      author={Dhruv Mahajan and Ross Girshick and Vignesh Ramanathan and Kaiming He and Manohar Paluri and Yixuan Li and Ashwin Bharambe and Laurens van der Maaten},
      year={2018},
      eprint={1805.00932},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: IG ResNeXt
  Paper:
    Title: Exploring the Limits of Weakly Supervised Pretraining
    URL: https://paperswithcode.com/paper/exploring-the-limits-of-weakly-supervised
Models:
- Name: ig_resnext101_32x16d
  In Collection: IG ResNeXt
  Metadata:
    FLOPs: 46623691776
    Parameters: 194030000
    File Size: 777518664
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - IG-3.5B-17k
    - ImageNet
    Training Resources: 336x GPUs
    ID: ig_resnext101_32x16d
    Epochs: 100
    Layers: 101
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 8064
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L874
  Weights: https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 84.16%
      Top 5 Accuracy: 97.19%
- Name: ig_resnext101_32x32d
  In Collection: IG ResNeXt
  Metadata:
    FLOPs: 112225170432
    Parameters: 468530000
    File Size: 1876573776
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - IG-3.5B-17k
    - ImageNet
    Training Resources: 336x GPUs
    ID: ig_resnext101_32x32d
    Epochs: 100
    Layers: 101
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 8064
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
    Minibatch Size: 8064
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L885
  Weights: https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 85.09%
      Top 5 Accuracy: 97.44%
- Name: ig_resnext101_32x48d
  In Collection: IG ResNeXt
  Metadata:
    FLOPs: 197446554624
    Parameters: 828410000
    File Size: 3317136976
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - IG-3.5B-17k
    - ImageNet
    Training Resources: 336x GPUs
    ID: ig_resnext101_32x48d
    Epochs: 100
    Layers: 101
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 8064
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L896
  Weights: https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 85.42%
      Top 5 Accuracy: 97.58%
- Name: ig_resnext101_32x8d
  In Collection: IG ResNeXt
  Metadata:
    FLOPs: 21180417024
    Parameters: 88790000
    File Size: 356056638
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - IG-3.5B-17k
    - ImageNet
    Training Resources: 336x GPUs
    ID: ig_resnext101_32x8d
    Epochs: 100
    Layers: 101
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 8064
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L863
  Weights: https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.7%
      Top 5 Accuracy: 96.64%
-->
