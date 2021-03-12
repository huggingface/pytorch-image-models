# Summary

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
Models:
- Name: ig_resnext101_32x32d
  Metadata:
    FLOPs: 112225170432
    Epochs: 100
    Batch Size: 8064
    Training Data:
    - IG-3.5B-17k
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 336x GPUs
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
    File Size: 1876573776
    Tasks:
    - Image Classification
    ID: ig_resnext101_32x32d
    Layers: 101
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
    Minibatch Size: 8064
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L885
  In Collection: IG ResNeXt
- Name: ig_resnext101_32x16d
  Metadata:
    FLOPs: 46623691776
    Epochs: 100
    Batch Size: 8064
    Training Data:
    - IG-3.5B-17k
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 336x GPUs
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
    File Size: 777518664
    Tasks:
    - Image Classification
    ID: ig_resnext101_32x16d
    Layers: 101
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L874
  In Collection: IG ResNeXt
- Name: ig_resnext101_32x48d
  Metadata:
    FLOPs: 197446554624
    Epochs: 100
    Batch Size: 8064
    Training Data:
    - IG-3.5B-17k
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 336x GPUs
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
    File Size: 3317136976
    Tasks:
    - Image Classification
    ID: ig_resnext101_32x48d
    Layers: 101
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L896
  In Collection: IG ResNeXt
- Name: ig_resnext101_32x8d
  Metadata:
    FLOPs: 21180417024
    Epochs: 100
    Batch Size: 8064
    Training Data:
    - IG-3.5B-17k
    - ImageNet
    Training Techniques:
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: 336x GPUs
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
    File Size: 356056638
    Tasks:
    - Image Classification
    ID: ig_resnext101_32x8d
    Layers: 101
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L863
  In Collection: IG ResNeXt
Collections:
- Name: IG ResNeXt
  Paper:
    title: Exploring the Limits of Weakly Supervised Pretraining
    url: https://paperswithcode.com//paper/exploring-the-limits-of-weakly-supervised
  type: model-index
Type: model-index
-->
