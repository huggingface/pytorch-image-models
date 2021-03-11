# Summary

**SE ResNeXt** is a variant of a [ResNext](https://www.paperswithcode.com/method/resnext) that employs [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) to enable the network to perform dynamic channel-wise feature recalibration.

The weights from this model were ported from Gluon.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{hu2019squeezeandexcitation,
      title={Squeeze-and-Excitation Networks}, 
      author={Jie Hu and Li Shen and Samuel Albanie and Gang Sun and Enhua Wu},
      year={2019},
      eprint={1709.01507},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: gluon_seresnext50_32x4d
  Metadata:
    FLOPs: 5475179184
    Training Data:
    - ImageNet
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
    - Squeeze-and-Excitation Block
    File Size: 110578827
    Tasks:
    - Image Classification
    ID: gluon_seresnext50_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L209
  In Collection: Gloun SEResNeXt
- Name: gluon_seresnext101_32x4d
  Metadata:
    FLOPs: 10302923504
    Training Data:
    - ImageNet
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
    - Squeeze-and-Excitation Block
    File Size: 196505510
    Tasks:
    - Image Classification
    ID: gluon_seresnext101_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L219
  In Collection: Gloun SEResNeXt
- Name: gluon_seresnext101_64x4d
  Metadata:
    FLOPs: 19958950640
    Training Data:
    - ImageNet
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
    - Squeeze-and-Excitation Block
    File Size: 353875948
    Tasks:
    - Image Classification
    ID: gluon_seresnext101_64x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L229
  In Collection: Gloun SEResNeXt
Collections:
- Name: Gloun SEResNeXt
  Paper:
    title: Squeeze-and-Excitation Networks
    url: https://papperswithcode.com//paper/squeeze-and-excitation-networks
  type: model-index
Type: model-index
-->
