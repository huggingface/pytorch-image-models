# Noisy Student (EfficientNet)

**Noisy Student Training** is a semi-supervised learning approach. It extends the idea of self-training
and distillation with the use of equal-or-larger student models and noise added to the student during learning. It has three main steps: 

1. train a teacher model on labeled images
2. use the teacher to generate pseudo labels on unlabeled images
3. train a student model on the combination of labeled images and pseudo labeled images. 

The algorithm is iterated a few times by treating the student as a teacher to relabel the unlabeled data and training a new student.

Noisy Student Training seeks to improve on self-training and distillation in two ways. First, it makes the student larger than, or at least equal to, the teacher so the student can better learn from a larger dataset. Second, it adds noise to the student so the noised student is forced to learn harder from the pseudo labels. To noise the student, it uses input noise such as RandAugment data augmentation, and model noise such as dropout and stochastic depth during training.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{xie2020selftraining,
      title={Self-training with Noisy Student improves ImageNet classification}, 
      author={Qizhe Xie and Minh-Thang Luong and Eduard Hovy and Quoc V. Le},
      year={2020},
      eprint={1911.04252},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!--
Models:
- Name: tf_efficientnet_b3_ns
  Metadata:
    FLOPs: 2275247568
    Epochs: 700
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 49385734
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b3_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.904'
    Momentum: 0.9
    Image Size: '300'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1457
  Config: ''
  In Collection: Noisy Student
- Name: tf_efficientnet_b1_ns
  Metadata:
    FLOPs: 883633200
    Epochs: 700
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 31516408
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b1_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.882'
    Momentum: 0.9
    Image Size: '240'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1437
  Config: ''
  In Collection: Noisy Student
- Name: tf_efficientnet_l2_ns
  Metadata:
    FLOPs: 611646113804
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 1925950424
    Tasks:
    - Image Classification
    Training Time: 6 days
    ID: tf_efficientnet_l2_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.96'
    Momentum: 0.9
    Image Size: '800'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1520
  Config: ''
  In Collection: Noisy Student
- Name: tf_efficientnet_b0_ns
  Metadata:
    FLOPs: 488688572
    Epochs: 700
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 21386709
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b0_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1427
  Config: ''
  In Collection: Noisy Student
- Name: tf_efficientnet_b2_ns
  Metadata:
    FLOPs: 1234321170
    Epochs: 700
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 36801803
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b2_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.89'
    Momentum: 0.9
    Image Size: '260'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1447
  Config: ''
  In Collection: Noisy Student
- Name: tf_efficientnet_b5_ns
  Metadata:
    FLOPs: 13176501888
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 122404944
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b5_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.934'
    Momentum: 0.9
    Image Size: '456'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1477
  Config: ''
  In Collection: Noisy Student
- Name: tf_efficientnet_b6_ns
  Metadata:
    FLOPs: 24180518488
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 173239537
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b6_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.942'
    Momentum: 0.9
    Image Size: '528'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1487
  Config: ''
  In Collection: Noisy Student
- Name: tf_efficientnet_b4_ns
  Metadata:
    FLOPs: 5749638672
    Epochs: 700
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 77995057
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b4_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.922'
    Momentum: 0.9
    Image Size: '380'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1467
  Config: ''
  In Collection: Noisy Student
- Name: tf_efficientnet_b7_ns
  Metadata:
    FLOPs: 48205304880
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Resources: Cloud TPU v3 Pod
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 266853140
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tf_efficientnet_b7_ns
    LR: 0.128
    Dropout: 0.5
    Crop Pct: '0.949'
    Momentum: 0.9
    Image Size: '600'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1498
  Config: ''
  In Collection: Noisy Student
Collections:
- Name: Noisy Student
  Paper:
    title: Self-training with Noisy Student improves ImageNet classification
    url: https://paperswithcode.com//paper/self-training-with-noisy-student-improves
  type: model-index
Type: model-index
-->
