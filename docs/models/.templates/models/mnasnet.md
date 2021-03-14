# MnasNet

**MnasNet** is a type of convolutional neural network optimized for mobile devices that is discovered through mobile neural architecture search, which explicitly incorporates model latency into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. The main building block is an [inverted residual block](https://paperswithcode.com/method/inverted-residual-block) (from [MobileNetV2](https://paperswithcode.com/method/mobilenetv2)).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{tan2019mnasnet,
      title={MnasNet: Platform-Aware Neural Architecture Search for Mobile}, 
      author={Mingxing Tan and Bo Chen and Ruoming Pang and Vijay Vasudevan and Mark Sandler and Andrew Howard and Quoc V. Le},
      year={2019},
      eprint={1807.11626},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: MNASNet
  Paper:
    Title: 'MnasNet: Platform-Aware Neural Architecture Search for Mobile'
    URL: https://paperswithcode.com/paper/mnasnet-platform-aware-neural-architecture
Models:
- Name: mnasnet_100
  In Collection: MNASNet
  Metadata:
    FLOPs: 416415488
    Parameters: 4380000
    File Size: 17731774
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Inverted Residual Block
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    ID: mnasnet_100
    Layers: 100
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4000
    Image Size: '224'
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L894
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.67%
      Top 5 Accuracy: 92.1%
- Name: semnasnet_100
  In Collection: MNASNet
  Metadata:
    FLOPs: 414570766
    Parameters: 3890000
    File Size: 15731489
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Inverted Residual Block
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: semnasnet_100
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L928
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.45%
      Top 5 Accuracy: 92.61%
-->
