# Summary

A **SENet** is a convolutional neural network architecture that employs [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) to enable the network to perform dynamic channel-wise feature recalibration.

The weights from this model were ported from [Gluon](https://cv.gluon.ai/model_zoo/classification.html).

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
- Name: gluon_senet154
  Metadata:
    FLOPs: 26681705136
    Training Data:
    - ImageNet
    Architecture:
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 461546622
    Tasks:
    - Image Classification
    ID: gluon_senet154
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L239
  In Collection: Gloun SENet
Collections:
- Name: Gloun SENet
  Paper:
    title: Squeeze-and-Excitation Networks
    url: https://paperswithcode.com//paper/squeeze-and-excitation-networks
  type: model-index
Type: model-index
-->
