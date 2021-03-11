# Summary

**Single-Path NAS** is a novel differentiable NAS method for designing hardware-efficient ConvNets in less than 4 hours.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{stamoulis2019singlepath,
      title={Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours}, 
      author={Dimitrios Stamoulis and Ruizhou Ding and Di Wang and Dimitrios Lymberopoulos and Bodhi Priyantha and Jie Liu and Diana Marculescu},
      year={2019},
      eprint={1904.02877},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!--
Models:
- Name: spnasnet_100
  Metadata:
    FLOPs: 442385600
    Training Data:
    - ImageNet
    Architecture:
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - ReLU
    File Size: 17902337
    Tasks:
    - Image Classification
    ID: spnasnet_100
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L995
  In Collection: SPNASNet
Collections:
- Name: SPNASNet
  Paper:
    title: 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4
      Hours'
    url: https://papperswithcode.com//paper/single-path-nas-designing-hardware-efficient
  type: model-index
Type: model-index
-->
