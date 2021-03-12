# Summary

**RegNetX** is a convolutional network design space with simple, regular models with parameters: depth $d$, initial width $w\_{0} > 0$, and slope $w\_{a} > 0$, and generates a different block width $u\_{j}$ for each block $j < d$. The key restriction for the RegNet types of model is that there is a linear parameterisation of block widths (the design space only contains models with this linear structure):

$$ u\_{j} = w\_{0} + w\_{a}\cdot{j} $$

For **RegNetX** we have additional restrictions: we set $b = 1$ (the bottleneck ratio), $12 \leq d \leq 28$, and $w\_{m} \geq 2$ (the width multiplier).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('regnetx_040', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `regnetx_040`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('regnetx_040', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
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
Models:
- Name: regnetx_040
  Metadata:
    FLOPs: 5095167744
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
    File Size: 88844824
    Tasks:
    - Image Classification
    ID: regnetx_040
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L373
  In Collection: RegNetX
- Name: regnetx_004
  Metadata:
    FLOPs: 510619136
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
    File Size: 20841309
    Tasks:
    - Image Classification
    ID: regnetx_004
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L343
  In Collection: RegNetX
- Name: regnetx_006
  Metadata:
    FLOPs: 771659136
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
    File Size: 24965172
    Tasks:
    - Image Classification
    ID: regnetx_006
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L349
  In Collection: RegNetX
- Name: regnetx_002
  Metadata:
    FLOPs: 255276032
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
    File Size: 10862199
    Tasks:
    - Image Classification
    ID: regnetx_002
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L337
  In Collection: RegNetX
- Name: regnetx_008
  Metadata:
    FLOPs: 1027038208
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
    File Size: 29235944
    Tasks:
    - Image Classification
    ID: regnetx_008
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L355
  In Collection: RegNetX
- Name: regnetx_016
  Metadata:
    FLOPs: 2059337856
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
    File Size: 36988158
    Tasks:
    - Image Classification
    ID: regnetx_016
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L361
  In Collection: RegNetX
- Name: regnetx_032
  Metadata:
    FLOPs: 4082555904
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
    File Size: 61509573
    Tasks:
    - Image Classification
    ID: regnetx_032
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L367
  In Collection: RegNetX
- Name: regnetx_064
  Metadata:
    FLOPs: 8303405824
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
    File Size: 105184854
    Tasks:
    - Image Classification
    ID: regnetx_064
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L379
  In Collection: RegNetX
- Name: regnetx_080
  Metadata:
    FLOPs: 10276726784
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
    File Size: 158720042
    Tasks:
    - Image Classification
    ID: regnetx_080
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L385
  In Collection: RegNetX
- Name: regnetx_120
  Metadata:
    FLOPs: 15536378368
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
    File Size: 184866342
    Tasks:
    - Image Classification
    ID: regnetx_120
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L391
  In Collection: RegNetX
- Name: regnetx_160
  Metadata:
    FLOPs: 20491740672
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
    File Size: 217623862
    Tasks:
    - Image Classification
    ID: regnetx_160
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L397
  In Collection: RegNetX
- Name: regnetx_320
  Metadata:
    FLOPs: 40798958592
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
    File Size: 431962133
    Tasks:
    - Image Classification
    ID: regnetx_320
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 5.0e-05
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/regnet.py#L403
  In Collection: RegNetX
Collections:
- Name: RegNetX
  Paper:
    title: Designing Network Design Spaces
    url: https://paperswithcode.com//paper/designing-network-design-spaces
  type: model-index
Type: model-index
-->