# NASNet

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
Type: model-index
Collections:
- Name: NASNet
  Paper:
    Title: Learning Transferable Architectures for Scalable Image Recognition
    URL: https://paperswithcode.com/paper/learning-transferable-architectures-for
Models:
- Name: nasnetalarge
  In Collection: NASNet
  Metadata:
    FLOPs: 30242402862
    Parameters: 88750000
    File Size: 356056626
    Architecture:
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 50x Tesla K40 GPUs
    ID: nasnetalarge
    Dropout: 0.5
    Crop Pct: '0.911'
    Momentum: 0.9
    Image Size: '331'
    Interpolation: bicubic
    Label Smoothing: 0.1
    RMSProp $\epsilon$: 1.0
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/nasnet.py#L562
  Weights: http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.63%
      Top 5 Accuracy: 96.05%
-->
