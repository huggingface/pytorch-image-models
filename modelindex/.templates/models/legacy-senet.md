# Summary

A **SENet** is a convolutional neural network architecture that employs [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) to enable the network to perform dynamic channel-wise feature recalibration.

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
- Name: legacy_senet154
  Metadata:
    FLOPs: 26659556016
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
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 461488402
    Tasks:
    - Image Classification
    ID: legacy_senet154
    LR: 0.6
    Layers: 154
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L440
  In Collection: Legacy SENet
Collections:
- Name: Legacy SENet
  Paper:
    title: Squeeze-and-Excitation Networks
    url: https://papperswithcode.com//paper/squeeze-and-excitation-networks
  type: model-index
Type: model-index
-->
