# Wide ResNet

**Wide Residual Networks** are a variant on [ResNets](https://paperswithcode.com/method/resnet) where we decrease depth and increase the width of residual networks. This is achieved through the use of [wide residual blocks](https://paperswithcode.com/method/wide-residual-block).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/ZagoruykoK16,
  author    = {Sergey Zagoruyko and
               Nikos Komodakis},
  title     = {Wide Residual Networks},
  journal   = {CoRR},
  volume    = {abs/1605.07146},
  year      = {2016},
  url       = {http://arxiv.org/abs/1605.07146},
  archivePrefix = {arXiv},
  eprint    = {1605.07146},
  timestamp = {Mon, 13 Aug 2018 16:46:42 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/ZagoruykoK16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: Wide ResNet
  Paper:
    Title: Wide Residual Networks
    URL: https://paperswithcode.com/paper/wide-residual-networks
Models:
- Name: wide_resnet101_2
  In Collection: Wide ResNet
  Metadata:
    FLOPs: 29304929280
    Parameters: 126890000
    File Size: 254695146
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Wide Residual Block
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: wide_resnet101_2
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/resnet.py#L802
  Weights: https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.85%
      Top 5 Accuracy: 94.28%
- Name: wide_resnet50_2
  In Collection: Wide ResNet
  Metadata:
    FLOPs: 14688058368
    Parameters: 68880000
    File Size: 275853271
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Wide Residual Block
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: wide_resnet50_2
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/resnet.py#L790
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.45%
      Top 5 Accuracy: 95.52%
-->
