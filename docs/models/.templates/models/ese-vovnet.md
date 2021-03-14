# ESE VoVNet

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
Models:
- Name: ese_vovnet39b
  Metadata:
    FLOPs: 9089259008
    Training Data:
    - ImageNet
    Architecture:
    - Batch Normalization
    - Convolution
    - Max Pooling
    - One-Shot Aggregation
    - ReLU
    File Size: 98397138
    Tasks:
    - Image Classification
    ID: ese_vovnet39b
    Layers: 39
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/vovnet.py#L371
  In Collection: ESE VovNet
- Name: ese_vovnet19b_dw
  Metadata:
    FLOPs: 1711959904
    Training Data:
    - ImageNet
    Architecture:
    - Batch Normalization
    - Convolution
    - Max Pooling
    - One-Shot Aggregation
    - ReLU
    File Size: 26243175
    Tasks:
    - Image Classification
    ID: ese_vovnet19b_dw
    Layers: 19
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/vovnet.py#L361
  In Collection: ESE VovNet
Collections:
- Name: ESE VovNet
  Paper:
    title: 'CenterMask : Real-Time Anchor-Free Instance Segmentation'
    url: https://paperswithcode.com//paper/centermask-real-time-anchor-free-instance-1
  type: model-index
Type: model-index
-->
