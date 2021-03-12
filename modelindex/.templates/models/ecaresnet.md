# Summary

An **ECA ResNet** is a variant on a [ResNet](https://paperswithcode.com/method/resnet) that utilises an [Efficient Channel Attention module](https://paperswithcode.com/method/efficient-channel-attention). Efficient Channel Attention is an architectural unit based on [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) that reduces model complexity without dimensionality reduction. 

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{wang2020ecanet,
      title={ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks}, 
      author={Qilong Wang and Banggu Wu and Pengfei Zhu and Peihua Li and Wangmeng Zuo and Qinghua Hu},
      year={2020},
      eprint={1910.03151},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: ecaresnet101d
  Metadata:
    FLOPs: 10377193728
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 4x RTX 2080Ti GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 178815067
    Tasks:
    - Image Classification
    ID: ecaresnet101d
    LR: 0.1
    Layers: 101
    Crop Pct: '0.875'
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1087
  In Collection: ECAResNet
- Name: ecaresnet101d_pruned
  Metadata:
    FLOPs: 4463972081
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 99852736
    Tasks:
    - Image Classification
    Training Time: ''
    ID: ecaresnet101d_pruned
    Layers: 101
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1097
  Config: ''
  In Collection: ECAResNet
- Name: ecaresnet50d_pruned
  Metadata:
    FLOPs: 3250730657
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 79990436
    Tasks:
    - Image Classification
    ID: ecaresnet50d_pruned
    Layers: 50
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1055
  In Collection: ECAResNet
- Name: ecaresnet50d
  Metadata:
    FLOPs: 5591090432
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 4x RTX 2080Ti GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 102579290
    Tasks:
    - Image Classification
    ID: ecaresnet50d
    LR: 0.1
    Layers: 50
    Crop Pct: '0.875'
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1045
  In Collection: ECAResNet
- Name: ecaresnetlight
  Metadata:
    FLOPs: 5276118784
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 120956612
    Tasks:
    - Image Classification
    ID: ecaresnetlight
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1077
  In Collection: ECAResNet
Collections:
- Name: ECAResNet
  Paper:
    title: 'ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks'
    url: https://paperswithcode.com//paper/eca-net-efficient-channel-attention-for-deep
  type: model-index
Type: model-index
-->
