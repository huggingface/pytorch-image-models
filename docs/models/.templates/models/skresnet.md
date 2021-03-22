# SK-ResNet

**SK ResNet** is a variant of a [ResNet](https://www.paperswithcode.com/method/resnet) that employs a [Selective Kernel](https://paperswithcode.com/method/selective-kernel) unit. In general, all the large kernel convolutions in the original bottleneck blocks in ResNet are replaced by the proposed [SK convolutions](https://paperswithcode.com/method/selective-kernel-convolution), enabling the network to choose appropriate receptive field sizes in an adaptive manner.

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
- Name: SKResNet
  Paper:
    Title: Selective Kernel Networks
    URL: https://paperswithcode.com/paper/selective-kernel-networks
Models:
- Name: skresnet18
  In Collection: SKResNet
  Metadata:
    FLOPs: 2333467136
    Parameters: 11960000
    File Size: 47923238
    Architecture:
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Residual Connection
    - Selective Kernel
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    ID: skresnet18
    LR: 0.1
    Epochs: 100
    Layers: 18
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/sknet.py#L148
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnet18_ra-4eec2804.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 73.03%
      Top 5 Accuracy: 91.17%
- Name: skresnet34
  In Collection: SKResNet
  Metadata:
    FLOPs: 4711849952
    Parameters: 22280000
    File Size: 89299314
    Architecture:
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Residual Connection
    - Selective Kernel
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    ID: skresnet34
    LR: 0.1
    Epochs: 100
    Layers: 34
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/sknet.py#L165
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnet34_ra-bdc0ccde.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.93%
      Top 5 Accuracy: 93.32%
-->
