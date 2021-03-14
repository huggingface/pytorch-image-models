# ESE-VoVNet

**VoVNet** is a convolutional neural network that seeks to make [DenseNet](https://paperswithcode.com/method/densenet) more efficient by concatenating all features only once in the last feature map, which makes input size constant and enables enlarging new output channel. 

Read about [one-shot aggregation here](https://paperswithcode.com/method/one-shot-aggregation).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{lee2019energy,
      title={An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection}, 
      author={Youngwan Lee and Joong-won Hwang and Sangrok Lee and Yuseok Bae and Jongyoul Park},
      year={2019},
      eprint={1904.09730},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: ESE VovNet
  Paper:
    Title: 'CenterMask : Real-Time Anchor-Free Instance Segmentation'
    URL: https://paperswithcode.com/paper/centermask-real-time-anchor-free-instance-1
Models:
- Name: ese_vovnet19b_dw
  In Collection: ESE VovNet
  Metadata:
    FLOPs: 1711959904
    Parameters: 6540000
    File Size: 26243175
    Architecture:
    - Batch Normalization
    - Convolution
    - Max Pooling
    - One-Shot Aggregation
    - ReLU
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: ese_vovnet19b_dw
    Layers: 19
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/vovnet.py#L361
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ese_vovnet19b_dw-a8741004.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.82%
      Top 5 Accuracy: 93.28%
- Name: ese_vovnet39b
  In Collection: ESE VovNet
  Metadata:
    FLOPs: 9089259008
    Parameters: 24570000
    File Size: 98397138
    Architecture:
    - Batch Normalization
    - Convolution
    - Max Pooling
    - One-Shot Aggregation
    - ReLU
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: ese_vovnet39b
    Layers: 39
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/vovnet.py#L371
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ese_vovnet39b-f912fe73.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.31%
      Top 5 Accuracy: 94.72%
-->
