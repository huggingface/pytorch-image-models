# SPNASNet

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
Type: model-index
Collections:
- Name: SPNASNet
  Paper:
    Title: 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4
      Hours'
    URL: https://paperswithcode.com/paper/single-path-nas-designing-hardware-efficient
Models:
- Name: spnasnet_100
  In Collection: SPNASNet
  Metadata:
    FLOPs: 442385600
    Parameters: 4420000
    File Size: 17902337
    Architecture:
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - ReLU
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: spnasnet_100
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L995
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.08%
      Top 5 Accuracy: 91.82%
-->
