# ResNeSt

A **ResNeSt** is a variant on a [ResNet](https://paperswithcode.com/method/resnet), which instead stacks [Split-Attention blocks](https://paperswithcode.com/method/split-attention). The cardinal group representations are then concatenated along the channel dimension: $V = \text{Concat}${$V^{1},V^{2},\cdots{V}^{K}$}. As in standard residual blocks, the final output $Y$ of otheur Split-Attention block is produced using a shortcut connection: $Y=V+X$, if the input and output feature-map share the same shape.  For blocks with a stride, an appropriate transformation $\mathcal{T}$ is applied to the shortcut connection to align the output shapes:  $Y=V+\mathcal{T}(X)$. For example, $\mathcal{T}$ can be strided convolution or combined convolution-with-pooling.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{zhang2020resnest,
      title={ResNeSt: Split-Attention Networks}, 
      author={Hang Zhang and Chongruo Wu and Zhongyue Zhang and Yi Zhu and Haibin Lin and Zhi Zhang and Yue Sun and Tong He and Jonas Mueller and R. Manmatha and Mu Li and Alexander Smola},
      year={2020},
      eprint={2004.08955},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: resnest50d_4s2x40d
  Metadata:
    FLOPs: 5657064720
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 122133282
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest50d_4s2x40d
    LR: 0.1
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L218
  Config: ''
  In Collection: ResNeSt
- Name: resnest200e
  Metadata:
    FLOPs: 45954387872
    Epochs: 270
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 193782911
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest200e
    LR: 0.1
    Layers: 200
    Dropout: 0.2
    Crop Pct: '0.909'
    Momentum: 0.9
    Image Size: '320'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L194
  Config: ''
  In Collection: ResNeSt
- Name: resnest14d
  Metadata:
    FLOPs: 3548594464
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 42562639
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest14d
    LR: 0.1
    Layers: 14
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L148
  Config: ''
  In Collection: ResNeSt
- Name: resnest101e
  Metadata:
    FLOPs: 17423183648
    Epochs: 270
    Batch Size: 4096
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 193782911
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest101e
    LR: 0.1
    Layers: 101
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '256'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L182
  Config: ''
  In Collection: ResNeSt
- Name: resnest269e
  Metadata:
    FLOPs: 100830307104
    Epochs: 270
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 445402691
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest269e
    LR: 0.1
    Layers: 269
    Dropout: 0.2
    Crop Pct: '0.928'
    Momentum: 0.9
    Image Size: '416'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L206
  Config: ''
  In Collection: ResNeSt
- Name: resnest26d
  Metadata:
    FLOPs: 4678918720
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 68470242
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest26d
    LR: 0.1
    Layers: 26
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L159
  Config: ''
  In Collection: ResNeSt
- Name: resnest50d
  Metadata:
    FLOPs: 6937106336
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 110273258
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest50d
    LR: 0.1
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L170
  Config: ''
  In Collection: ResNeSt
- Name: resnest50d_1s4x24d
  Metadata:
    FLOPs: 5686764544
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 103045531
    Tasks:
    - Image Classification
    ID: resnest50d_1s4x24d
    LR: 0.1
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L229
  In Collection: ResNeSt
Collections:
- Name: ResNeSt
  Paper:
    title: 'ResNeSt: Split-Attention Networks'
    url: https://paperswithcode.com//paper/resnest-split-attention-networks
  type: model-index
Type: model-index
-->
