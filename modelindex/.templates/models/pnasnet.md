# Summary

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
Models:
- Name: pnasnet5large
  Metadata:
    FLOPs: 31458865950
    Batch Size: 1600
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - RMSProp
    - Weight Decay
    Training Resources: 100x NVIDIA P100 GPUs
    Architecture:
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - ReLU
    File Size: 345153926
    Tasks:
    - Image Classification
    ID: pnasnet5large
    LR: 0.015
    Dropout: 0.5
    Crop Pct: '0.911'
    Momentum: 0.9
    Image Size: '331'
    Interpolation: bicubic
    Label Smoothing: 0.1
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/pnasnet.py#L343
  In Collection: PNASNet
Collections:
- Name: PNASNet
  Paper:
    title: Progressive Neural Architecture Search
    url: https://papperswithcode.com//paper/progressive-neural-architecture-search
  type: model-index
Type: model-index
-->
