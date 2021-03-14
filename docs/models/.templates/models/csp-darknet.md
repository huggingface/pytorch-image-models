# CSP-DarkNet

**CSPDarknet53** is a convolutional neural network and backbone for object detection that uses [DarkNet-53](https://paperswithcode.com/method/darknet-53). It employs a CSPNet strategy to partition the feature map of the base layer into two parts and then merges them through a cross-stage hierarchy. The use of a split and merge strategy allows for more gradient flow through the network. 

This CNN is used as the backbone for [YOLOv4](https://paperswithcode.com/method/yolov4).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection}, 
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: CSP DarkNet
  Paper:
    Title: 'YOLOv4: Optimal Speed and Accuracy of Object Detection'
    URL: https://paperswithcode.com/paper/yolov4-optimal-speed-and-accuracy-of-object
Models:
- Name: cspdarknet53
  In Collection: CSP DarkNet
  Metadata:
    FLOPs: 8545018880
    Parameters: 27640000
    File Size: 110775135
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Mish
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - CutMix
    - Label Smoothing
    - Mosaic
    - Polynomial Learning Rate Decay
    - SGD with Momentum
    - Self-Adversarial Training
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 1x NVIDIA RTX 2070 GPU
    ID: cspdarknet53
    LR: 0.1
    Layers: 53
    Crop Pct: '0.887'
    Momentum: 0.9
    Batch Size: 128
    Image Size: '256'
    Warmup Steps: 1000
    Weight Decay: 0.0005
    Interpolation: bilinear
    Training Steps: 8000000
    FPS (GPU RTX 2070): 66
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/cspnet.py#L441
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspdarknet53_ra_256-d05c7c21.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.05%
      Top 5 Accuracy: 95.09%
-->
