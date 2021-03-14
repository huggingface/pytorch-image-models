# CSP-ResNet

**CSPResNet** is a convolutional neural network where we apply the Cross Stage Partial Network (CSPNet) approach to [ResNet](https://paperswithcode.com/method/resnet). The CSPNet partitions the feature map of the base layer into two parts and then merges them through a cross-stage hierarchy. The use of a split and merge strategy allows for more gradient flow through the network.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{wang2019cspnet,
      title={CSPNet: A New Backbone that can Enhance Learning Capability of CNN}, 
      author={Chien-Yao Wang and Hong-Yuan Mark Liao and I-Hau Yeh and Yueh-Hua Wu and Ping-Yang Chen and Jun-Wei Hsieh},
      year={2019},
      eprint={1911.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: CSP ResNet
  Paper:
    Title: 'CSPNet: A New Backbone that can Enhance Learning Capability of CNN'
    URL: https://paperswithcode.com/paper/cspnet-a-new-backbone-that-can-enhance
Models:
- Name: cspresnet50
  In Collection: CSP ResNet
  Metadata:
    FLOPs: 5924992000
    Parameters: 21620000
    File Size: 86679303
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
    - Label Smoothing
    - Polynomial Learning Rate Decay
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: cspresnet50
    LR: 0.1
    Layers: 50
    Crop Pct: '0.887'
    Momentum: 0.9
    Batch Size: 128
    Image Size: '256'
    Weight Decay: 0.005
    Interpolation: bilinear
    Training Steps: 8000000
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/cspnet.py#L415
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspresnet50_ra-d3e8d487.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.57%
      Top 5 Accuracy: 94.71%
-->
