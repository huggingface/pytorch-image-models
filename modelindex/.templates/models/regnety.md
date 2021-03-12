# Summary

**RegNetY** is a convolutional network design space with simple, regular models with parameters: depth $d$, initial width $w\_{0} > 0$, and slope $w\_{a} > 0$, and generates a different block width $u\_{j}$ for each block $j < d$. The key restriction for the RegNet types of model is that there is a linear parameterisation of block widths (the design space only contains models with this linear structure):

$$ u\_{j} = w\_{0} + w\_{a}\cdot{j} $$

For **RegNetX** authors have additional restrictions: we set $b = 1$ (the bottleneck ratio), $12 \leq d \leq 28$, and $w\_{m} \geq 2$ (the width multiplier).

For **RegNetY** authors make one change, which is to include [Squeeze-and-Excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{radosavovic2020designing,
      title={Designing Network Design Spaces}, 
      author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
      year={2020},
      eprint={2003.13678},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: regnety_002
  Metadata:
    FLOPs: 255754236
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 12782926
    Tasks:
    - Image Classification
    ID: regnety_002
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L409
  In Collection: RegNetY
- Name: regnety_016
  Metadata:
    FLOPs: 2070895094
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 45115589
    Tasks:
    - Image Classification
    ID: regnety_016
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L433
  In Collection: RegNetY
- Name: regnety_004
  Metadata:
    FLOPs: 515664568
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 17542753
    Tasks:
    - Image Classification
    ID: regnety_004
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L415
  In Collection: RegNetY
- Name: regnety_006
  Metadata:
    FLOPs: 771746928
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 24394127
    Tasks:
    - Image Classification
    ID: regnety_006
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L421
  In Collection: RegNetY
- Name: regnety_008
  Metadata:
    FLOPs: 1023448952
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 25223268
    Tasks:
    - Image Classification
    ID: regnety_008
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L427
  In Collection: RegNetY
- Name: regnety_032
  Metadata:
    FLOPs: 4081118714
    Epochs: 100
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 78084523
    Tasks:
    - Image Classification
    ID: regnety_032
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L439
  In Collection: RegNetY
- Name: regnety_080
  Metadata:
    FLOPs: 10233621420
    Epochs: 100
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 157124671
    Tasks:
    - Image Classification
    ID: regnety_080
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L457
  In Collection: RegNetY
- Name: regnety_040
  Metadata:
    FLOPs: 5105933432
    Epochs: 100
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 82913909
    Tasks:
    - Image Classification
    ID: regnety_040
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L445
  In Collection: RegNetY
- Name: regnety_064
  Metadata:
    FLOPs: 8167730444
    Epochs: 100
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 122751416
    Tasks:
    - Image Classification
    ID: regnety_064
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L451
  In Collection: RegNetY
- Name: regnety_120
  Metadata:
    FLOPs: 15542094856
    Epochs: 100
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 207743949
    Tasks:
    - Image Classification
    ID: regnety_120
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L463
  In Collection: RegNetY
- Name: regnety_160
  Metadata:
    FLOPs: 20450196852
    Epochs: 100
    Batch Size: 512
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 334916722
    Tasks:
    - Image Classification
    ID: regnety_160
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L469
  In Collection: RegNetY
- Name: regnety_320
  Metadata:
    FLOPs: 41492618394
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    File Size: 580891965
    Tasks:
    - Image Classification
    ID: regnety_320
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L475
  In Collection: RegNetY
Collections:
- Name: RegNetY
  Paper:
    title: Designing Network Design Spaces
    url: https://paperswithcode.com//paper/designing-network-design-spaces
  type: model-index
Type: model-index
-->
