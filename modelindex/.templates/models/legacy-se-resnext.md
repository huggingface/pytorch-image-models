# Summary

**SE ResNeXt** is a variant of a [ResNeXt](https://www.paperswithcode.com/method/resnext) that employs [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) to enable the network to perform dynamic channel-wise feature recalibration.

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
- Name: legacy_seresnext101_32x4d
  Metadata:
    FLOPs: 10287698672
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA Titan X GPUs
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
    File Size: 196466866
    Tasks:
    - Image Classification
    ID: legacy_seresnext101_32x4d
    LR: 0.6
    Layers: 101
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L462
  In Collection: Legacy SE ResNeXt
- Name: legacy_seresnext26_32x4d
  Metadata:
    FLOPs: 3187342304
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA Titan X GPUs
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
    File Size: 67346327
    Tasks:
    - Image Classification
    ID: legacy_seresnext26_32x4d
    LR: 0.6
    Layers: 26
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L448
  In Collection: Legacy SE ResNeXt
- Name: legacy_seresnext50_32x4d
  Metadata:
    FLOPs: 5459954352
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA Titan X GPUs
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
    File Size: 110559176
    Tasks:
    - Image Classification
    ID: legacy_seresnext50_32x4d
    LR: 0.6
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L455
  In Collection: Legacy SE ResNeXt
Collections:
- Name: Legacy SE ResNeXt
  Paper:
    title: Squeeze-and-Excitation Networks
    url: https://papperswithcode.com//paper/squeeze-and-excitation-networks
  type: model-index
Type: model-index
-->
