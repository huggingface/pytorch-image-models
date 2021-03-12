# Summary

**CSPResNeXt** is a convolutional neural network where we apply the Cross Stage Partial Network (CSPNet) approach to [ResNeXt](https://paperswithcode.com/method/resnext). The CSPNet partitions the feature map of the base layer into two parts and then merges them through a cross-stage hierarchy. The use of a split and merge strategy allows for more gradient flow through the network.

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
- Name: cspresnext50
  Metadata:
    FLOPs: 3962945536
    Batch Size: 128
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - Polynomial Learning Rate Decay
    - SGD with Momentum
    - Weight Decay
    Training Resources: 1x GPU
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    File Size: 82562887
    Tasks:
    - Image Classification
    Training Time: ''
    ID: cspresnext50
    LR: 0.1
    Layers: 50
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.005
    Interpolation: bilinear
    Training Steps: 8000000
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/cspnet.py#L430
  Config: ''
  In Collection: CSP ResNeXt
Collections:
- Name: CSP ResNeXt
  Paper:
    title: 'CSPNet: A New Backbone that can Enhance Learning Capability of CNN'
    url: https://paperswithcode.com//paper/cspnet-a-new-backbone-that-can-enhance
  type: model-index
Type: model-index
-->
