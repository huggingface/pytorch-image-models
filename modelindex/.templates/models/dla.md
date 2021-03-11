# Summary

Extending  “shallow” skip connections, **Dense Layer Aggregation (DLA)** incorporates more depth and sharing. The authors introduce two structures for deep layer aggregation (DLA): iterative deep aggregation (IDA) and hierarchical deep aggregation (HDA). These structures are expressed through an architectural framework, independent of the choice of backbone, for compatibility with current and future networks. 

IDA focuses on fusing resolutions and scales while HDA focuses on merging features from all modules and channels. IDA follows the base hierarchy to refine resolution and aggregate scale stage-bystage. HDA assembles its own hierarchy of tree-structured connections that cross and merge stages to aggregate different levels of representation. 

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{yu2019deep,
      title={Deep Layer Aggregation}, 
      author={Fisher Yu and Dequan Wang and Evan Shelhamer and Trevor Darrell},
      year={2019},
      eprint={1707.06484},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: dla60
  Metadata:
    FLOPs: 4256251880
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 89560235
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60
    LR: 0.1
    Layers: 60
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L394
  Config: ''
  In Collection: DLA
- Name: dla46_c
  Metadata:
    FLOPs: 583277288
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 5307963
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla46_c
    LR: 0.1
    Layers: 46
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L369
  Config: ''
  In Collection: DLA
- Name: dla102x2
  Metadata:
    FLOPs: 9343847400
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 167645295
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla102x2
    LR: 0.1
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L426
  Config: ''
  In Collection: DLA
- Name: dla102
  Metadata:
    FLOPs: 7192952808
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 135290579
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla102
    LR: 0.1
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L410
  Config: ''
  In Collection: DLA
- Name: dla102x
  Metadata:
    FLOPs: 5886821352
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 107552695
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla102x
    LR: 0.1
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L418
  Config: ''
  In Collection: DLA
- Name: dla169
  Metadata:
    FLOPs: 11598004200
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 216547113
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla169
    LR: 0.1
    Layers: 169
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L434
  Config: ''
  In Collection: DLA
- Name: dla46x_c
  Metadata:
    FLOPs: 544052200
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 4387641
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla46x_c
    LR: 0.1
    Layers: 46
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L378
  Config: ''
  In Collection: DLA
- Name: dla60_res2net
  Metadata:
    FLOPs: 4147578504
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 84886593
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60_res2net
    Layers: 60
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L346
  Config: ''
  In Collection: DLA
- Name: dla60_res2next
  Metadata:
    FLOPs: 3485335272
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 69639245
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60_res2next
    Layers: 60
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L354
  Config: ''
  In Collection: DLA
- Name: dla34
  Metadata:
    FLOPs: 3070105576
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 63228658
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla34
    LR: 0.1
    Layers: 32
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L362
  Config: ''
  In Collection: DLA
- Name: dla60x
  Metadata:
    FLOPs: 3544204264
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 70883139
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60x
    LR: 0.1
    Layers: 60
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L402
  Config: ''
  In Collection: DLA
- Name: dla60x_c
  Metadata:
    FLOPs: 593325032
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 5454396
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60x_c
    LR: 0.1
    Layers: 60
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L386
  Config: ''
  In Collection: DLA
Collections:
- Name: DLA
  Paper:
    title: Deep Layer Aggregation
    url: https://papperswithcode.com//paper/deep-layer-aggregation
  type: model-index
Type: model-index
-->
