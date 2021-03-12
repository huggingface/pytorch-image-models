# Summary

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
Models:
- Name: cspresnet50
  Metadata:
    FLOPs: 5924992000
    Batch Size: 128
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - Polynomial Learning Rate Decay
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
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
    File Size: 86679303
    Tasks:
    - Image Classification
    Training Time: ''
    ID: cspresnet50
    LR: 0.1
    Layers: 50
    Crop Pct: '0.887'
    Momentum: 0.9
    Image Size: '256'
    Weight Decay: 0.005
    Interpolation: bilinear
    Training Steps: 8000000
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/cspnet.py#L415
  Config: ''
  In Collection: CSP ResNet
Collections:
- Name: CSP ResNet
  Paper:
    title: 'CSPNet: A New Backbone that can Enhance Learning Capability of CNN'
    url: https://paperswithcode.com//paper/cspnet-a-new-backbone-that-can-enhance
  type: model-index
Type: model-index
-->
