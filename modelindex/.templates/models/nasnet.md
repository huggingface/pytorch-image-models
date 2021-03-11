# Summary

**NASNet** is a type of convolutional neural network discovered through neural architecture search. The building blocks consist of normal and reduction cells.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{zoph2018learning,
      title={Learning Transferable Architectures for Scalable Image Recognition}, 
      author={Barret Zoph and Vijay Vasudevan and Jonathon Shlens and Quoc V. Le},
      year={2018},
      eprint={1707.07012},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: nasnetalarge
  Metadata:
    FLOPs: 30242402862
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - RMSProp
    - Weight Decay
    Training Resources: 50x Tesla K40 GPUs
    Architecture:
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - ReLU
    File Size: 356056626
    Tasks:
    - Image Classification
    Training Time: ''
    ID: nasnetalarge
    Dropout: 0.5
    Crop Pct: '0.911'
    Momentum: 0.9
    Image Size: '331'
    Interpolation: bicubic
    Label Smoothing: 0.1
    RMSProp $\epsilon$: 1.0
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/nasnet.py#L562
  Config: ''
  In Collection: NASNet
Collections:
- Name: NASNet
  Paper:
    title: Learning Transferable Architectures for Scalable Image Recognition
    url: https://papperswithcode.com//paper/learning-transferable-architectures-for
  type: model-index
Type: model-index
-->
