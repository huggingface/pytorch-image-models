# FBNet

**FBNet** is a type of convolutional neural architectures discovered through [DNAS](https://paperswithcode.com/method/dnas) neural architecture search. It utilises a basic type of image model block inspired by [MobileNetv2](https://paperswithcode.com/method/mobilenetv2) that utilises depthwise convolutions and an inverted residual structure (see components).

The principal building block is the [FBNet Block](https://paperswithcode.com/method/fbnet-block).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{wu2019fbnet,
      title={FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search}, 
      author={Bichen Wu and Xiaoliang Dai and Peizhao Zhang and Yanghan Wang and Fei Sun and Yiming Wu and Yuandong Tian and Peter Vajda and Yangqing Jia and Kurt Keutzer},
      year={2019},
      eprint={1812.03443},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: FBNet
  Paper:
    Title: 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural
      Architecture Search'
    URL: https://paperswithcode.com/paper/fbnet-hardware-aware-efficient-convnet-design
Models:
- Name: fbnetc_100
  In Collection: FBNet
  Metadata:
    FLOPs: 508940064
    Parameters: 5570000
    File Size: 22525094
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Dropout
    - FBNet Block
    - Global Average Pooling
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    ID: fbnetc_100
    LR: 0.1
    Epochs: 360
    Layers: 22
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0005
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L985
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.12%
      Top 5 Accuracy: 92.37%
-->
