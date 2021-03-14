# SWSL ResNet

**Residual Networks**, or **ResNets**, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping. They stack [residual blocks](https://paperswithcode.com/method/residual-block) ontop of each other to form network: e.g. a ResNet-50 has fifty layers using these blocks. 

The models in this collection utilise semi-weakly supervised learning to improve the performance of the model. The approach brings important gains to standard architectures for image, video and fine-grained classification. 

Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/abs-1905-00546,
  author    = {I. Zeki Yalniz and
               Herv{\'{e}} J{\'{e}}gou and
               Kan Chen and
               Manohar Paluri and
               Dhruv Mahajan},
  title     = {Billion-scale semi-supervised learning for image classification},
  journal   = {CoRR},
  volume    = {abs/1905.00546},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.00546},
  archivePrefix = {arXiv},
  eprint    = {1905.00546},
  timestamp = {Mon, 28 Sep 2020 08:19:37 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1905-00546.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: SWSL ResNet
  Paper:
    Title: Billion-scale semi-supervised learning for image classification
    URL: https://paperswithcode.com/paper/billion-scale-semi-supervised-learning-for
Models:
- Name: swsl_resnet18
  In Collection: SWSL ResNet
  Metadata:
    FLOPs: 2337073152
    Parameters: 11690000
    File Size: 46811375
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
    - IG-1B-Targeted
    - ImageNet
    Training Resources: 64x GPUs
    ID: swsl_resnet18
    LR: 0.0015
    Epochs: 30
    Layers: 18
    Crop Pct: '0.875'
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L954
  Weights: https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 73.28%
      Top 5 Accuracy: 91.76%
- Name: swsl_resnet50
  In Collection: SWSL ResNet
  Metadata:
    FLOPs: 5282531328
    Parameters: 25560000
    File Size: 102480594
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
    - IG-1B-Targeted
    - ImageNet
    Training Resources: 64x GPUs
    ID: swsl_resnet50
    LR: 0.0015
    Epochs: 30
    Layers: 50
    Crop Pct: '0.875'
    Batch Size: 1536
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L965
  Weights: https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.14%
      Top 5 Accuracy: 95.97%
-->
