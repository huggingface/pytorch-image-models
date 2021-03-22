# PNASNet

**Progressive Neural Architecture Search**, or **PNAS**, is a method for learning the structure of convolutional neural networks (CNNs). It uses a sequential model-based optimization (SMBO) strategy, where we search the space of cell structures, starting with simple (shallow) models and progressing to complex ones, pruning out unpromising structures as we go. 

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{liu2018progressive,
      title={Progressive Neural Architecture Search}, 
      author={Chenxi Liu and Barret Zoph and Maxim Neumann and Jonathon Shlens and Wei Hua and Li-Jia Li and Li Fei-Fei and Alan Yuille and Jonathan Huang and Kevin Murphy},
      year={2018},
      eprint={1712.00559},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: PNASNet
  Paper:
    Title: Progressive Neural Architecture Search
    URL: https://paperswithcode.com/paper/progressive-neural-architecture-search
Models:
- Name: pnasnet5large
  In Collection: PNASNet
  Metadata:
    FLOPs: 31458865950
    Parameters: 86060000
    File Size: 345153926
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
    Training Resources: 100x NVIDIA P100 GPUs
    ID: pnasnet5large
    LR: 0.015
    Dropout: 0.5
    Crop Pct: '0.911'
    Momentum: 0.9
    Batch Size: 1600
    Image Size: '331'
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/pnasnet.py#L343
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/pnasnet5large-bf079911.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 0.98%
      Top 5 Accuracy: 18.58%
-->
