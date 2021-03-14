# (Gluon) Xception

**Xception** is a convolutional neural network architecture that relies solely on [depthwise separable convolution](https://paperswithcode.com/method/depthwise-separable-convolution) layers.

The weights from this model were ported from [Gluon](https://cv.gluon.ai/model_zoo/classification.html).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{chollet2017xception,
      title={Xception: Deep Learning with Depthwise Separable Convolutions}, 
      author={François Chollet},
      year={2017},
      eprint={1610.02357},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: Gloun Xception
  Paper:
    Title: 'Xception: Deep Learning with Depthwise Separable Convolutions'
    URL: https://paperswithcode.com/paper/xception-deep-learning-with-depthwise
Models:
- Name: gluon_xception65
  In Collection: Gloun Xception
  Metadata:
    FLOPs: 17594889728
    Parameters: 39920000
    File Size: 160551306
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_xception65
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_xception.py#L241
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_xception-7015a15c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.7%
      Top 5 Accuracy: 94.87%
-->
