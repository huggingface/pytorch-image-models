# Summary

**Big Transfer (BiT)** is a type of pretraining recipe that pre-trains  on a large supervised source dataset, and fine-tunes the weights on the target task. Models are trained on the JFT-300M dataset. The finetuned models contained in this collection are finetuned on ImageNet.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{kolesnikov2020big,
      title={Big Transfer (BiT): General Visual Representation Learning}, 
      author={Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Joan Puigcerver and Jessica Yung and Sylvain Gelly and Neil Houlsby},
      year={2020},
      eprint={1912.11370},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: resnetv2_152x4_bitm
  Metadata:
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 3746270104
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_152x4_bitm
    Crop Pct: '1.0'
    Image Size: '480'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L465
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_152x2_bitm
  Metadata:
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 945476668
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_152x2_bitm
    Crop Pct: '1.0'
    Image Size: '480'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L458
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_50x1_bitm
  Metadata:
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 102242668
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_50x1_bitm
    LR: 0.03
    Layers: 50
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L430
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_101x3_bitm
  Metadata:
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 1551830100
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_101x3_bitm
    LR: 0.03
    Layers: 101
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L451
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_50x3_bitm
  Metadata:
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 869321580
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_50x3_bitm
    LR: 0.03
    Layers: 50
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L437
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_101x1_bitm
  Metadata:
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 178256468
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_101x1_bitm
    LR: 0.03
    Layers: 101
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L444
  Config: ''
  In Collection: Big Transfer
Collections:
- Name: Big Transfer
  Paper:
    title: 'Big Transfer (BiT): General Visual Representation Learning'
    url: https://papperswithcode.com//paper/large-scale-learning-of-general-visual
  type: model-index
Type: model-index
-->
