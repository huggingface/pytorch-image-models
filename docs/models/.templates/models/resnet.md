# ResNet

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
Type: model-index
Collections:
- Name: ResNet
  Paper:
    Title: Deep Residual Learning for Image Recognition
    URL: https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition
Models:
- Name: resnet18
  In Collection: ResNet
  Metadata:
    FLOPs: 2337073152
    Parameters: 11690000
    File Size: 46827520
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: resnet18
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L641
  Weights: https://download.pytorch.org/models/resnet18-5c106cde.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 69.74%
      Top 5 Accuracy: 89.09%
- Name: resnet26
  In Collection: ResNet
  Metadata:
    FLOPs: 3026804736
    Parameters: 16000000
    File Size: 64129972
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: resnet26
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L675
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.29%
      Top 5 Accuracy: 92.57%
- Name: resnet34
  In Collection: ResNet
  Metadata:
    FLOPs: 4718469120
    Parameters: 21800000
    File Size: 87290831
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: resnet34
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L658
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.11%
      Top 5 Accuracy: 92.28%
- Name: resnet50
  In Collection: ResNet
  Metadata:
    FLOPs: 5282531328
    Parameters: 25560000
    File Size: 102488165
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: resnet50
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L691
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.04%
      Top 5 Accuracy: 94.39%
- Name: resnetblur50
  In Collection: ResNet
  Metadata:
    FLOPs: 6621606912
    Parameters: 25560000
    File Size: 102488165
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
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: resnetblur50
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L1160
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnetblur50-84f4748f.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.29%
      Top 5 Accuracy: 94.64%
- Name: tv_resnet101
  In Collection: ResNet
  Metadata:
    FLOPs: 10068547584
    Parameters: 44550000
    File Size: 178728960
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tv_resnet101
    LR: 0.1
    Epochs: 90
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Batch Size: 32
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L761
  Weights: https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.37%
      Top 5 Accuracy: 93.56%
- Name: tv_resnet152
  In Collection: ResNet
  Metadata:
    FLOPs: 14857660416
    Parameters: 60190000
    File Size: 241530880
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tv_resnet152
    LR: 0.1
    Epochs: 90
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Batch Size: 32
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L769
  Weights: https://download.pytorch.org/models/resnet152-b121ed2d.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.32%
      Top 5 Accuracy: 94.05%
- Name: tv_resnet34
  In Collection: ResNet
  Metadata:
    FLOPs: 4718469120
    Parameters: 21800000
    File Size: 87306240
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tv_resnet34
    LR: 0.1
    Epochs: 90
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Batch Size: 32
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L745
  Weights: https://download.pytorch.org/models/resnet34-333f7ec4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 73.3%
      Top 5 Accuracy: 91.42%
- Name: tv_resnet50
  In Collection: ResNet
  Metadata:
    FLOPs: 5282531328
    Parameters: 25560000
    File Size: 102502400
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tv_resnet50
    LR: 0.1
    Epochs: 90
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Batch Size: 32
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L753
  Weights: https://download.pytorch.org/models/resnet50-19c8e357.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.16%
      Top 5 Accuracy: 92.88%
-->
