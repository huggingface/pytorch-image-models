# RegNetY

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
Type: model-index
Collections:
- Name: RegNetY
  Paper:
    Title: Designing Network Design Spaces
    URL: https://paperswithcode.com/paper/designing-network-design-spaces
Models:
- Name: regnety_002
  In Collection: RegNetY
  Metadata:
    FLOPs: 255754236
    Parameters: 3160000
    File Size: 12782926
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_002
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L409
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_002-e68ca334.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 70.28%
      Top 5 Accuracy: 89.55%
- Name: regnety_004
  In Collection: RegNetY
  Metadata:
    FLOPs: 515664568
    Parameters: 4340000
    File Size: 17542753
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_004
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L415
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_004-0db870e6.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.02%
      Top 5 Accuracy: 91.76%
- Name: regnety_006
  In Collection: RegNetY
  Metadata:
    FLOPs: 771746928
    Parameters: 6060000
    File Size: 24394127
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_006
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L421
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_006-c67e57ec.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.27%
      Top 5 Accuracy: 92.53%
- Name: regnety_008
  In Collection: RegNetY
  Metadata:
    FLOPs: 1023448952
    Parameters: 6260000
    File Size: 25223268
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_008
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L427
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_008-dc900dbe.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.32%
      Top 5 Accuracy: 93.07%
- Name: regnety_016
  In Collection: RegNetY
  Metadata:
    FLOPs: 2070895094
    Parameters: 11200000
    File Size: 45115589
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_016
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L433
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_016-54367f74.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.87%
      Top 5 Accuracy: 93.73%
- Name: regnety_032
  In Collection: RegNetY
  Metadata:
    FLOPs: 4081118714
    Parameters: 19440000
    File Size: 78084523
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_032
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L439
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/regnety_032_ra-7f2439f9.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.01%
      Top 5 Accuracy: 95.91%
- Name: regnety_040
  In Collection: RegNetY
  Metadata:
    FLOPs: 5105933432
    Parameters: 20650000
    File Size: 82913909
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_040
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L445
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_040-f0d569f9.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.23%
      Top 5 Accuracy: 94.64%
- Name: regnety_064
  In Collection: RegNetY
  Metadata:
    FLOPs: 8167730444
    Parameters: 30580000
    File Size: 122751416
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_064
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L451
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_064-0a48325c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.73%
      Top 5 Accuracy: 94.76%
- Name: regnety_080
  In Collection: RegNetY
  Metadata:
    FLOPs: 10233621420
    Parameters: 39180000
    File Size: 157124671
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_080
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L457
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_080-e7f3eb93.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.87%
      Top 5 Accuracy: 94.83%
- Name: regnety_120
  In Collection: RegNetY
  Metadata:
    FLOPs: 15542094856
    Parameters: 51820000
    File Size: 207743949
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_120
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L463
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_120-721ba79a.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.38%
      Top 5 Accuracy: 95.12%
- Name: regnety_160
  In Collection: RegNetY
  Metadata:
    FLOPs: 20450196852
    Parameters: 83590000
    File Size: 334916722
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_160
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L469
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_160-d64013cd.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.28%
      Top 5 Accuracy: 94.97%
- Name: regnety_320
  In Collection: RegNetY
  Metadata:
    FLOPs: 41492618394
    Parameters: 145050000
    File Size: 580891965
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnety_320
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L475
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_320-ba464b29.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.8%
      Top 5 Accuracy: 95.25%
-->
