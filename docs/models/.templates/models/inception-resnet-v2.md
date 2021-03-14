# Inception ResNet v2

**Inception-ResNet-v2** is a convolutional neural architecture that builds on the Inception family of architectures but incorporates [residual connections](https://paperswithcode.com/method/residual-connection) (replacing the filter concatenation stage of the Inception architecture).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{szegedy2016inceptionv4,
      title={Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning}, 
      author={Christian Szegedy and Sergey Ioffe and Vincent Vanhoucke and Alex Alemi},
      year={2016},
      eprint={1602.07261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: Inception ResNet v2
  Paper:
    Title: Inception-v4, Inception-ResNet and the Impact of Residual Connections on
      Learning
    URL: https://paperswithcode.com/paper/inception-v4-inception-resnet-and-the-impact
Models:
- Name: inception_resnet_v2
  In Collection: Inception ResNet v2
  Metadata:
    FLOPs: 16959133120
    Parameters: 55850000
    File Size: 223774238
    Architecture:
    - Average Pooling
    - Dropout
    - Inception-ResNet-v2 Reduction-B
    - Inception-ResNet-v2-A
    - Inception-ResNet-v2-B
    - Inception-ResNet-v2-C
    - Reduction-A
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 20x NVIDIA Kepler GPUs
    ID: inception_resnet_v2
    LR: 0.045
    Dropout: 0.2
    Crop Pct: '0.897'
    Momentum: 0.9
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/inception_resnet_v2.py#L343
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/inception_resnet_v2-940b1cd6.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 0.95%
      Top 5 Accuracy: 17.29%
-->
