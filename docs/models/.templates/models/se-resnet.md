# SE-ResNet

**SE ResNet** is a variant of a [ResNet](https://www.paperswithcode.com/method/resnet) that employs [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) to enable the network to perform dynamic channel-wise feature recalibration.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{hu2019squeezeandexcitation,
      title={Squeeze-and-Excitation Networks}, 
      author={Jie Hu and Li Shen and Samuel Albanie and Gang Sun and Enhua Wu},
      year={2019},
      eprint={1709.01507},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: SE ResNet
  Paper:
    Title: Squeeze-and-Excitation Networks
    URL: https://paperswithcode.com/paper/squeeze-and-excitation-networks
Models:
- Name: seresnet152d
  In Collection: SE ResNet
  Metadata:
    FLOPs: 20161904304
    Parameters: 66840000
    File Size: 268144497
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: seresnet152d
    LR: 0.6
    Epochs: 100
    Layers: 152
    Dropout: 0.2
    Crop Pct: '0.94'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '256'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1206
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet152d_ra2-04464dd2.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 83.74%
      Top 5 Accuracy: 96.77%
- Name: seresnet50
  In Collection: SE ResNet
  Metadata:
    FLOPs: 5285062320
    Parameters: 28090000
    File Size: 112621903
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: seresnet50
    LR: 0.6
    Epochs: 100
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1180
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.26%
      Top 5 Accuracy: 95.07%
-->
