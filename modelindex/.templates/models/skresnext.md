# Summary

**SK ResNeXt** is a variant of a [ResNeXt](https://www.paperswithcode.com/method/resnext) that employs a [Selective Kernel](https://paperswithcode.com/method/selective-kernel) unit. In general, all the large kernel convolutions in the original bottleneck blocks in ResNext are replaced by the proposed [SK convolutions](https://paperswithcode.com/method/selective-kernel-convolution), enabling the network to choose appropriate receptive field sizes in an adaptive manner.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{li2019selective,
      title={Selective Kernel Networks}, 
      author={Xiang Li and Wenhai Wang and Xiaolin Hu and Jian Yang},
      year={2019},
      eprint={1903.06586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: skresnext50_32x4d
  Metadata:
    FLOPs: 5739845824
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    Architecture:
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - Residual Connection
    - Selective Kernel
    - Softmax
    File Size: 110340975
    Tasks:
    - Image Classification
    ID: skresnext50_32x4d
    LR: 0.1
    Layers: 50
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/sknet.py#L210
  In Collection: SKResNeXt
Collections:
- Name: SKResNeXt
  Paper:
    title: Selective Kernel Networks
    url: https://paperswithcode.com//paper/selective-kernel-networks
  type: model-index
Type: model-index
-->
