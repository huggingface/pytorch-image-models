# RegNetX

**RegNetX** is a convolutional network design space with simple, regular models with parameters: depth $d$, initial width $w\_{0} > 0$, and slope $w\_{a} > 0$, and generates a different block width $u\_{j}$ for each block $j < d$. The key restriction for the RegNet types of model is that there is a linear parameterisation of block widths (the design space only contains models with this linear structure):

$$ u\_{j} = w\_{0} + w\_{a}\cdot{j} $$

For **RegNetX** we have additional restrictions: we set $b = 1$ (the bottleneck ratio), $12 \leq d \leq 28$, and $w\_{m} \geq 2$ (the width multiplier).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('regnetx_002', pretrained=True)
model.eval()
```

To load and preprocess the image:
```python 
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension
```

To get the model predictions:
```python
import torch
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
# prints: torch.Size([1000])
```

To get the top-5 predictions class names:
```python
# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
# prints class names and probabilities like:
# [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]
```

Replace the model name with the variant you want to use, e.g. `regnetx_002`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('regnetx_002', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

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
- Name: RegNetX
  Paper:
    Title: Designing Network Design Spaces
    URL: https://paperswithcode.com/paper/designing-network-design-spaces
Models:
- Name: regnetx_002
  In Collection: RegNetX
  Metadata:
    FLOPs: 255276032
    Parameters: 2680000
    File Size: 10862199
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_002
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L337
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_002-e7e85e5c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 68.75%
      Top 5 Accuracy: 88.56%
- Name: regnetx_004
  In Collection: RegNetX
  Metadata:
    FLOPs: 510619136
    Parameters: 5160000
    File Size: 20841309
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_004
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L343
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_004-7d0e9424.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 72.39%
      Top 5 Accuracy: 90.82%
- Name: regnetx_006
  In Collection: RegNetX
  Metadata:
    FLOPs: 771659136
    Parameters: 6200000
    File Size: 24965172
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_006
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L349
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_006-85ec1baa.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 73.84%
      Top 5 Accuracy: 91.68%
- Name: regnetx_008
  In Collection: RegNetX
  Metadata:
    FLOPs: 1027038208
    Parameters: 7260000
    File Size: 29235944
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_008
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L355
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_008-d8b470eb.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.05%
      Top 5 Accuracy: 92.34%
- Name: regnetx_016
  In Collection: RegNetX
  Metadata:
    FLOPs: 2059337856
    Parameters: 9190000
    File Size: 36988158
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_016
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L361
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_016-65ca972a.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.95%
      Top 5 Accuracy: 93.43%
- Name: regnetx_032
  In Collection: RegNetX
  Metadata:
    FLOPs: 4082555904
    Parameters: 15300000
    File Size: 61509573
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_032
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L367
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_032-ed0c7f7e.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.15%
      Top 5 Accuracy: 94.09%
- Name: regnetx_040
  In Collection: RegNetX
  Metadata:
    FLOPs: 5095167744
    Parameters: 22120000
    File Size: 88844824
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_040
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L373
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_040-73c2a654.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.48%
      Top 5 Accuracy: 94.25%
- Name: regnetx_064
  In Collection: RegNetX
  Metadata:
    FLOPs: 8303405824
    Parameters: 26210000
    File Size: 105184854
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_064
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L379
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_064-29278baa.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.06%
      Top 5 Accuracy: 94.47%
- Name: regnetx_080
  In Collection: RegNetX
  Metadata:
    FLOPs: 10276726784
    Parameters: 39570000
    File Size: 158720042
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_080
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L385
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_080-7c7fcab1.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.21%
      Top 5 Accuracy: 94.55%
- Name: regnetx_120
  In Collection: RegNetX
  Metadata:
    FLOPs: 15536378368
    Parameters: 46110000
    File Size: 184866342
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_120
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L391
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_120-65d5521e.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.61%
      Top 5 Accuracy: 94.73%
- Name: regnetx_160
  In Collection: RegNetX
  Metadata:
    FLOPs: 20491740672
    Parameters: 54280000
    File Size: 217623862
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_160
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L397
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_160-c98c4112.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.84%
      Top 5 Accuracy: 94.82%
- Name: regnetx_320
  In Collection: RegNetX
  Metadata:
    FLOPs: 40798958592
    Parameters: 107810000
    File Size: 431962133
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Grouped Convolution
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA V100 GPUs
    ID: regnetx_320
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L403
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_320-8ea38b93.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.25%
      Top 5 Accuracy: 95.03%
-->