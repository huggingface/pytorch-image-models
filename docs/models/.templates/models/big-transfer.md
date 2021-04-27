# Big Transfer (BiT)

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
Type: model-index
Collections:
- Name: Big Transfer
  Paper:
    Title: 'Big Transfer (BiT): General Visual Representation Learning'
    URL: https://paperswithcode.com/paper/large-scale-learning-of-general-visual
Models:
- Name: resnetv2_101x1_bitm
  In Collection: Big Transfer
  Metadata:
    FLOPs: 5330896
    Parameters: 44540000
    File Size: 178256468
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
    Tasks:
    - Image Classification
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: Cloud TPUv3-512
    ID: resnetv2_101x1_bitm
    LR: 0.03
    Epochs: 90
    Layers: 101
    Crop Pct: '1.0'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L444
  Weights: https://storage.googleapis.com/bit_models/BiT-M-R101x1-ILSVRC2012.npz
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.21%
      Top 5 Accuracy: 96.47%
- Name: resnetv2_101x3_bitm
  In Collection: Big Transfer
  Metadata:
    FLOPs: 15988688
    Parameters: 387930000
    File Size: 1551830100
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
    Tasks:
    - Image Classification
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: Cloud TPUv3-512
    ID: resnetv2_101x3_bitm
    LR: 0.03
    Epochs: 90
    Layers: 101
    Crop Pct: '1.0'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L451
  Weights: https://storage.googleapis.com/bit_models/BiT-M-R101x3-ILSVRC2012.npz
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 84.38%
      Top 5 Accuracy: 97.37%
- Name: resnetv2_152x2_bitm
  In Collection: Big Transfer
  Metadata:
    FLOPs: 10659792
    Parameters: 236340000
    File Size: 945476668
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
    Tasks:
    - Image Classification
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    - JFT-300M
    ID: resnetv2_152x2_bitm
    Crop Pct: '1.0'
    Image Size: '480'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L458
  Weights: https://storage.googleapis.com/bit_models/BiT-M-R152x2-ILSVRC2012.npz
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 84.4%
      Top 5 Accuracy: 97.43%
- Name: resnetv2_152x4_bitm
  In Collection: Big Transfer
  Metadata:
    FLOPs: 21317584
    Parameters: 936530000
    File Size: 3746270104
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
    Tasks:
    - Image Classification
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: Cloud TPUv3-512
    ID: resnetv2_152x4_bitm
    Crop Pct: '1.0'
    Image Size: '480'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L465
  Weights: https://storage.googleapis.com/bit_models/BiT-M-R152x4-ILSVRC2012.npz
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 84.95%
      Top 5 Accuracy: 97.45%
- Name: resnetv2_50x1_bitm
  In Collection: Big Transfer
  Metadata:
    FLOPs: 5330896
    Parameters: 25550000
    File Size: 102242668
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
    Tasks:
    - Image Classification
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: Cloud TPUv3-512
    ID: resnetv2_50x1_bitm
    LR: 0.03
    Epochs: 90
    Layers: 50
    Crop Pct: '1.0'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L430
  Weights: https://storage.googleapis.com/bit_models/BiT-M-R50x1-ILSVRC2012.npz
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.19%
      Top 5 Accuracy: 95.63%
- Name: resnetv2_50x3_bitm
  In Collection: Big Transfer
  Metadata:
    FLOPs: 15988688
    Parameters: 217320000
    File Size: 869321580
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
    Tasks:
    - Image Classification
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: Cloud TPUv3-512
    ID: resnetv2_50x3_bitm
    LR: 0.03
    Epochs: 90
    Layers: 50
    Crop Pct: '1.0'
    Momentum: 0.9
    Batch Size: 4096
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L437
  Weights: https://storage.googleapis.com/bit_models/BiT-M-R50x3-ILSVRC2012.npz
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 83.75%
      Top 5 Accuracy: 97.12%
-->
