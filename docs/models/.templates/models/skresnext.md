# SK-ResNeXt

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
Type: model-index
Collections:
- Name: SKResNeXt
  Paper:
    Title: Selective Kernel Networks
    URL: https://paperswithcode.com/paper/selective-kernel-networks
Models:
- Name: skresnext50_32x4d
  In Collection: SKResNeXt
  Metadata:
    FLOPs: 5739845824
    Parameters: 27480000
    File Size: 110340975
    Architecture:
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - Residual Connection
    - Selective Kernel
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    ID: skresnext50_32x4d
    LR: 0.1
    Epochs: 100
    Layers: 50
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/sknet.py#L210
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnext50_ra-f40e40bf.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.15%
      Top 5 Accuracy: 94.64%
-->
