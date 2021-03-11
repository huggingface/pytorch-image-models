# Summary

**Xception** is a convolutional neural network architecture that relies solely on [depthwise separable convolution](https://paperswithcode.com/method/depthwise-separable-convolution) layers. The weights from this model were ported from Gluon.

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
Models:
- Name: gluon_xception65
  Metadata:
    FLOPs: 17594889728
    Training Data:
    - ImageNet
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
    File Size: 160551306
    Tasks:
    - Image Classification
    ID: gluon_xception65
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_xception.py#L241
  In Collection: Gloun Xception
Collections:
- Name: Gloun Xception
  Paper:
    title: 'Xception: Deep Learning with Depthwise Separable Convolutions'
    url: https://papperswithcode.com//paper/xception-deep-learning-with-depthwise
  type: model-index
Type: model-index
-->
