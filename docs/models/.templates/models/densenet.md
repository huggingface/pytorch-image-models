# DenseNet

**DenseNet** is a type of convolutional neural network that utilises dense connections between layers, through [Dense Blocks](http://www.paperswithcode.com/method/dense-block), where we connect *all layers* (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers.

The **DenseNet Blur** variant in this collection by Ross Wightman employs [Blur Pooling](http://www.paperswithcode.com/method/blur-pooling)

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/HuangLW16a,
  author    = {Gao Huang and
               Zhuang Liu and
               Kilian Q. Weinberger},
  title     = {Densely Connected Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1608.06993},
  year      = {2016},
  url       = {http://arxiv.org/abs/1608.06993},
  archivePrefix = {arXiv},
  eprint    = {1608.06993},
  timestamp = {Mon, 10 Sep 2018 15:49:32 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HuangLW16a.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

<!--
Type: model-index
Collections:
- Name: DenseNet
  Paper:
    Title: Densely Connected Convolutional Networks
    URL: https://paperswithcode.com/paper/densely-connected-convolutional-networks
Models:
- Name: densenet121
  In Collection: DenseNet
  Metadata:
    FLOPs: 3641843200
    Parameters: 7980000
    File Size: 32376726
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Kaiming Initialization
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    ID: densenet121
    LR: 0.1
    Epochs: 90
    Layers: 121
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L295
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/densenet121_ra-50efcf5c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.56%
      Top 5 Accuracy: 92.65%
- Name: densenet161
  In Collection: DenseNet
  Metadata:
    FLOPs: 9931959264
    Parameters: 28680000
    File Size: 115730790
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Kaiming Initialization
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    ID: densenet161
    LR: 0.1
    Epochs: 90
    Layers: 161
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L347
  Weights: https://download.pytorch.org/models/densenet161-8d451a50.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.36%
      Top 5 Accuracy: 93.63%
- Name: densenet169
  In Collection: DenseNet
  Metadata:
    FLOPs: 4316945792
    Parameters: 14150000
    File Size: 57365526
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Kaiming Initialization
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    ID: densenet169
    LR: 0.1
    Epochs: 90
    Layers: 169
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L327
  Weights: https://download.pytorch.org/models/densenet169-b2777c0a.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.9%
      Top 5 Accuracy: 93.02%
- Name: densenet201
  In Collection: DenseNet
  Metadata:
    FLOPs: 5514321024
    Parameters: 20010000
    File Size: 81131730
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Kaiming Initialization
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Data:
    - ImageNet
    ID: densenet201
    LR: 0.1
    Epochs: 90
    Layers: 201
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L337
  Weights: https://download.pytorch.org/models/densenet201-c1103571.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.29%
      Top 5 Accuracy: 93.48%
- Name: densenetblur121d
  In Collection: DenseNet
  Metadata:
    FLOPs: 3947812864
    Parameters: 8000000
    File Size: 32456500
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Blur Pooling
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: densenetblur121d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L305
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/densenetblur121d_ra-100dcfbc.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.59%
      Top 5 Accuracy: 93.2%
- Name: tv_densenet121
  In Collection: DenseNet
  Metadata:
    FLOPs: 3641843200
    Parameters: 7980000
    File Size: 32342954
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tv_densenet121
    LR: 0.1
    Epochs: 90
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Batch Size: 32
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L379
  Weights: https://download.pytorch.org/models/densenet121-a639ec97.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.74%
      Top 5 Accuracy: 92.15%
-->
