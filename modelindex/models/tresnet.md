# Summary

A **TResNet** is a variant on a [ResNet](https://paperswithcode.com/method/resnet) that aim to boost accuracy while maintaining GPU training and inference efficiency.  They contain several design tricks including a SpaceToDepth stem, [Anti-Alias downsampling](https://paperswithcode.com/method/anti-alias-downsampling), In-Place Activated BatchNorm, Blocks selection and [squeeze-and-excitation layers](https://paperswithcode.com/method/squeeze-and-excitation-block).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('tresnet_l', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `tresnet_l`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('tresnet_l', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{ridnik2020tresnet,
      title={TResNet: High Performance GPU-Dedicated Architecture}, 
      author={Tal Ridnik and Hussam Lawen and Asaf Noy and Emanuel Ben Baruch and Gilad Sharir and Itamar Friedman},
      year={2020},
      eprint={2003.13630},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: tresnet_l
  Metadata:
    FLOPs: 10873416792
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    File Size: 224440219
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_l
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L267
  Config: ''
  In Collection: TResNet
- Name: tresnet_l_448
  Metadata:
    FLOPs: 43488238584
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    File Size: 224440219
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_l_448
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L285
  Config: ''
  In Collection: TResNet
- Name: tresnet_m
  Metadata:
    FLOPs: 5733048064
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    File Size: 125861314
    Tasks:
    - Image Classification
    Training Time: < 24 hours
    ID: tresnet_m
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L261
  Config: ''
  In Collection: TResNet
- Name: tresnet_m_448
  Metadata:
    FLOPs: 22929743104
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    File Size: 125861314
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_m_448
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L279
  Config: ''
  In Collection: TResNet
- Name: tresnet_xl
  Metadata:
    FLOPs: 15162534034
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    File Size: 314378965
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_xl
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L273
  Config: ''
  In Collection: TResNet
- Name: tresnet_xl_448
  Metadata:
    FLOPs: 60641712730
    Epochs: 300
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - Cutout
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA 100 GPUs
    Architecture:
    - 1x1 Convolution
    - Anti-Alias Downsampling
    - Convolution
    - Global Average Pooling
    - InPlace-ABN
    - Leaky ReLU
    - ReLU
    - Residual Connection
    - Squeeze-and-Excitation Block
    File Size: 224440219
    Tasks:
    - Image Classification
    Training Time: ''
    ID: tresnet_xl_448
    LR: 0.01
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '448'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/tresnet.py#L291
  Config: ''
  In Collection: TResNet
Collections:
- Name: TResNet
  Paper:
    title: 'TResNet: High Performance GPU-Dedicated Architecture'
    url: https://paperswithcode.com//paper/tresnet-high-performance-gpu-dedicated
  type: model-index
Type: model-index
-->