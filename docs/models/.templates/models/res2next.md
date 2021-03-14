# Res2NeXt

**Res2NeXt** is an image model that employs a variation on [ResNeXt](https://paperswithcode.com/method/resnext) bottleneck residual blocks. The motivation is to be able to represent features at multiple scales. This is achieved through a novel building block for CNNs that constructs hierarchical residual-like connections within one single residual block. This represents multi-scale features at a granular level and increases the range of receptive fields for each network layer.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{Gao_2021,
   title={Res2Net: A New Multi-Scale Backbone Architecture},
   volume={43},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/TPAMI.2019.2938758},
   DOI={10.1109/tpami.2019.2938758},
   number={2},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
   year={2021},
   month={Feb},
   pages={652–662}
}
```

<!--
Type: model-index
Collections:
- Name: Res2NeXt
  Paper:
    Title: 'Res2Net: A New Multi-scale Backbone Architecture'
    URL: https://paperswithcode.com/paper/res2net-a-new-multi-scale-backbone
Models:
- Name: res2next50
  In Collection: Res2NeXt
  Metadata:
    FLOPs: 5396798208
    Parameters: 24670000
    File Size: 99019592
    Architecture:
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - ReLU
    - Res2NeXt Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x Titan Xp GPUs
    ID: res2next50
    LR: 0.1
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/res2net.py#L207
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2next50_4s-6ef7e7bf.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.24%
      Top 5 Accuracy: 93.91%
-->
