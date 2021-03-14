# (Gluon) ResNet

**Residual Networks**, or **ResNets**, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping. They stack [residual blocks](https://paperswithcode.com/method/residual-block) ontop of each other to form network: e.g. a ResNet-50 has fifty layers using these blocks. 

The weights from this model were ported from [Gluon](https://cv.gluon.ai/model_zoo/classification.html).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('gluon_resnet101_v1b', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `gluon_resnet101_v1b`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('gluon_resnet101_v1b', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1512.03385},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03385},
  archivePrefix = {arXiv},
  eprint    = {1512.03385},
  timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: Gloun ResNet
  Paper:
    Title: Deep Residual Learning for Image Recognition
    URL: https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition
Models:
- Name: gluon_resnet101_v1b
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 10068547584
    Parameters: 44550000
    File Size: 178723172
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet101_v1b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L89
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1b-3b017079.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.3%
      Top 5 Accuracy: 94.53%
- Name: gluon_resnet101_v1c
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 10376567296
    Parameters: 44570000
    File Size: 178802575
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet101_v1c
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L113
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1c-1f26822a.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.53%
      Top 5 Accuracy: 94.59%
- Name: gluon_resnet101_v1d
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 10377018880
    Parameters: 44570000
    File Size: 178802755
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet101_v1d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L138
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1d-0f9c8644.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.4%
      Top 5 Accuracy: 95.02%
- Name: gluon_resnet101_v1s
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 11805511680
    Parameters: 44670000
    File Size: 179221777
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet101_v1s
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L166
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1s-60fe0cc1.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.29%
      Top 5 Accuracy: 95.16%
- Name: gluon_resnet152_v1b
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 14857660416
    Parameters: 60190000
    File Size: 241534001
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet152_v1b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L97
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1b-c1edb0dd.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.69%
      Top 5 Accuracy: 94.73%
- Name: gluon_resnet152_v1c
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 15165680128
    Parameters: 60210000
    File Size: 241613404
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet152_v1c
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L121
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1c-a3bb0b98.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.91%
      Top 5 Accuracy: 94.85%
- Name: gluon_resnet152_v1d
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 15166131712
    Parameters: 60210000
    File Size: 241613584
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet152_v1d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L147
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1d-bd354e12.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.48%
      Top 5 Accuracy: 95.2%
- Name: gluon_resnet152_v1s
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 16594624512
    Parameters: 60320000
    File Size: 242032606
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet152_v1s
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L175
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1s-dcc41b81.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.02%
      Top 5 Accuracy: 95.42%
- Name: gluon_resnet18_v1b
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 2337073152
    Parameters: 11690000
    File Size: 46816736
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet18_v1b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L65
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet18_v1b-0757602b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 70.84%
      Top 5 Accuracy: 89.76%
- Name: gluon_resnet34_v1b
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 4718469120
    Parameters: 21800000
    File Size: 87295112
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet34_v1b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L73
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet34_v1b-c6d82d59.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.59%
      Top 5 Accuracy: 92.0%
- Name: gluon_resnet50_v1b
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 5282531328
    Parameters: 25560000
    File Size: 102493763
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet50_v1b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L81
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1b-0ebe02e2.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.58%
      Top 5 Accuracy: 93.72%
- Name: gluon_resnet50_v1c
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 5590551040
    Parameters: 25580000
    File Size: 102573166
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet50_v1c
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L105
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1c-48092f55.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.01%
      Top 5 Accuracy: 93.99%
- Name: gluon_resnet50_v1d
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 5591002624
    Parameters: 25580000
    File Size: 102573346
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet50_v1d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L129
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1d-818a1b1b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.06%
      Top 5 Accuracy: 94.46%
- Name: gluon_resnet50_v1s
  In Collection: Gloun ResNet
  Metadata:
    FLOPs: 7019495424
    Parameters: 25680000
    File Size: 102992368
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_resnet50_v1s
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_resnet.py#L156
  Weights: https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1s-1762acc0.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.7%
      Top 5 Accuracy: 94.25%
-->