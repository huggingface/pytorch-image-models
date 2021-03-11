# Summary

**Xception** is a convolutional neural network architecture that relies solely on [depthwise separable convolution layers](https://paperswithcode.com/method/depthwise-separable-convolution).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/ZagoruykoK16,
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
- Name: xception
  Metadata:
    FLOPs: 10600506792
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
    File Size: 91675053
    Tasks:
    - Image Classification
    ID: xception
    Crop Pct: '0.897'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/xception.py#L229
  In Collection: Xception
- Name: xception41
  Metadata:
    FLOPs: 11681983232
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
    File Size: 108422028
    Tasks:
    - Image Classification
    ID: xception41
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/xception_aligned.py#L181
  In Collection: Xception
- Name: xception65
  Metadata:
    FLOPs: 17585702144
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
    File Size: 160536780
    Tasks:
    - Image Classification
    ID: xception65
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/xception_aligned.py#L200
  In Collection: Xception
- Name: xception71
  Metadata:
    FLOPs: 22817346560
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
    File Size: 170295556
    Tasks:
    - Image Classification
    ID: xception71
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/xception_aligned.py#L219
  In Collection: Xception
Collections:
- Name: Xception
  Paper:
    title: 'Xception: Deep Learning with Depthwise Separable Convolutions'
    url: https://papperswithcode.com//paper/xception-deep-learning-with-depthwise
  type: model-index
Type: model-index
-->
