# Summary

**Residual Networks**, or **ResNets**, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping. They stack [residual blocks](https://paperswithcode.com/method/residual-block) ontop of each other to form network: e.g. a ResNet-50 has fifty layers using these blocks. 

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1512.03385},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03385},
  archivePrefix = {arXiv},
  eprint    = {1512.03385},
  timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Models:
- Name: resnet26
  Metadata:
    FLOPs: 3026804736
    Training Data:
    - ImageNet
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
    File Size: 64129972
    Tasks:
    - Image Classification
    ID: resnet26
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L675
  In Collection: ResNet
- Name: tv_resnet152
  Metadata:
    FLOPs: 14857660416
    Epochs: 90
    Batch Size: 32
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
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
    File Size: 241530880
    Tasks:
    - Image Classification
    ID: tv_resnet152
    LR: 0.1
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L769
  In Collection: ResNet
- Name: resnet18
  Metadata:
    FLOPs: 2337073152
    Training Data:
    - ImageNet
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
    File Size: 46827520
    Tasks:
    - Image Classification
    ID: resnet18
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L641
  In Collection: ResNet
- Name: resnet50
  Metadata:
    FLOPs: 5282531328
    Training Data:
    - ImageNet
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
    File Size: 102488165
    Tasks:
    - Image Classification
    ID: resnet50
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L691
  In Collection: ResNet
- Name: resnet34
  Metadata:
    FLOPs: 4718469120
    Training Data:
    - ImageNet
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
    File Size: 87290831
    Tasks:
    - Image Classification
    ID: resnet34
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L658
  In Collection: ResNet
- Name: resnetblur50
  Metadata:
    FLOPs: 6621606912
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Blur Pooling
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 102488165
    Tasks:
    - Image Classification
    ID: resnetblur50
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L1160
  In Collection: ResNet
- Name: tv_resnet34
  Metadata:
    FLOPs: 4718469120
    Epochs: 90
    Batch Size: 32
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
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
    File Size: 87306240
    Tasks:
    - Image Classification
    ID: tv_resnet34
    LR: 0.1
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L745
  In Collection: ResNet
- Name: tv_resnet101
  Metadata:
    FLOPs: 10068547584
    Epochs: 90
    Batch Size: 32
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
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
    File Size: 178728960
    Tasks:
    - Image Classification
    ID: tv_resnet101
    LR: 0.1
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L761
  In Collection: ResNet
- Name: tv_resnet50
  Metadata:
    FLOPs: 5282531328
    Epochs: 90
    Batch Size: 32
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
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
    File Size: 102502400
    Tasks:
    - Image Classification
    ID: tv_resnet50
    LR: 0.1
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L753
  In Collection: ResNet
Collections:
- Name: ResNet
  Paper:
    title: Deep Residual Learning for Image Recognition
    url: https://paperswithcode.com//paper/deep-residual-learning-for-image-recognition
  type: model-index
Type: model-index
-->
