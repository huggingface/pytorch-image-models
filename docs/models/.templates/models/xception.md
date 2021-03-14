# Xception

**Xception** is a convolutional neural network architecture that relies solely on [depthwise separable convolution layers](https://paperswithcode.com/method/depthwise-separable-convolution).

The weights from this model were ported from [Tensorflow/Models](https://github.com/tensorflow/models).

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
Type: model-index
Collections:
- Name: Xception
  Paper:
    Title: 'Xception: Deep Learning with Depthwise Separable Convolutions'
    URL: https://paperswithcode.com/paper/xception-deep-learning-with-depthwise
Models:
- Name: xception
  In Collection: Xception
  Metadata:
    FLOPs: 10600506792
    Parameters: 22860000
    File Size: 91675053
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
    ID: xception
    Crop Pct: '0.897'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/xception.py#L229
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.05%
      Top 5 Accuracy: 94.4%
- Name: xception41
  In Collection: Xception
  Metadata:
    FLOPs: 11681983232
    Parameters: 26970000
    File Size: 108422028
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
    ID: xception41
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/xception_aligned.py#L181
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_41-e6439c97.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.54%
      Top 5 Accuracy: 94.28%
- Name: xception65
  In Collection: Xception
  Metadata:
    FLOPs: 17585702144
    Parameters: 39920000
    File Size: 160536780
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
    ID: xception65
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/xception_aligned.py#L200
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_65-c9ae96e8.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.55%
      Top 5 Accuracy: 94.66%
- Name: xception71
  In Collection: Xception
  Metadata:
    FLOPs: 22817346560
    Parameters: 42340000
    File Size: 170295556
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
    ID: xception71
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/xception_aligned.py#L219
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_71-8eec7df1.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.88%
      Top 5 Accuracy: 94.93%
-->
