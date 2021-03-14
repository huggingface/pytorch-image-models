# Inception v4

**Inception-v4** is a convolutional neural network architecture that builds on previous iterations of the Inception family by simplifying the architecture and using more inception modules than [Inception-v3](https://paperswithcode.com/method/inception-v3).
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
- Name: Inception v4
  Paper:
    Title: Inception-v4, Inception-ResNet and the Impact of Residual Connections on
      Learning
    URL: https://paperswithcode.com/paper/inception-v4-inception-resnet-and-the-impact
Models:
- Name: inception_v4
  In Collection: Inception v4
  Metadata:
    FLOPs: 15806527936
    Parameters: 42680000
    File Size: 171082495
    Architecture:
    - Average Pooling
    - Dropout
    - Inception-A
    - Inception-B
    - Inception-C
    - Reduction-A
    - Reduction-B
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
    ID: inception_v4
    LR: 0.045
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/inception_v4.py#L313
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/inceptionv4-8e4777a0.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 1.01%
      Top 5 Accuracy: 16.85%
-->
