# Summary

**Res2Net** is an image model that employs a variation on [ResNeXt](https://paperswithcode.com/method/resnext) bottleneck residual blocks. The motivation is to be able to represent features at multiple scales. This is achieved through a novel building block for CNNs that constructs hierarchical residual-like connections within one single residual block. This represents multi-scale features at a granular level and increases the range of receptive fields for each network layer.

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
Models:
- Name: res2next50
  Metadata:
    FLOPs: 5396798208
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 4x Titan Xp GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - ReLU
    - Res2NeXt Block
    File Size: 99019592
    Tasks:
    - Image Classification
    ID: res2next50
    LR: 0.1
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/res2net.py#L207
  In Collection: Res2NeXt
Collections:
- Name: Res2NeXt
  Paper:
    title: 'Res2Net: A New Multi-scale Backbone Architecture'
    url: https://papperswithcode.com//paper/res2net-a-new-multi-scale-backbone
  type: model-index
Type: model-index
-->
