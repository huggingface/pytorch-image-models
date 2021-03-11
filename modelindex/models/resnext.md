# Summary

A **ResNeXt** repeats a [building block](https://paperswithcode.com/method/resnext-block) that aggregates a set of transformations with the same topology. Compared to a [ResNet](https://paperswithcode.com/method/resnet), it exposes a new dimension,  *cardinality* (the size of the set of transformations) $C$, as an essential factor in addition to the dimensions of depth and width. 

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('resnext101_32x8d', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `resnext101_32x8d`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('resnext101_32x8d', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/XieGDTH16,
  author    = {Saining Xie and
               Ross B. Girshick and
               Piotr Doll{\'{a}}r and
               Zhuowen Tu and
               Kaiming He},
  title     = {Aggregated Residual Transformations for Deep Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1611.05431},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.05431},
  archivePrefix = {arXiv},
  eprint    = {1611.05431},
  timestamp = {Mon, 13 Aug 2018 16:45:58 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/XieGDTH16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Models:
- Name: resnext101_32x8d
  Metadata:
    FLOPs: 21180417024
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    File Size: 356082095
    Tasks:
    - Image Classification
    ID: resnext101_32x8d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L877
  In Collection: ResNeXt
- Name: resnext50_32x4d
  Metadata:
    FLOPs: 5472648192
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    File Size: 100435887
    Tasks:
    - Image Classification
    ID: resnext50_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L851
  In Collection: ResNeXt
- Name: tv_resnext50_32x4d
  Metadata:
    FLOPs: 5472648192
    Epochs: 90
    Batch Size: 32
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    File Size: 100441675
    Tasks:
    - Image Classification
    ID: tv_resnext50_32x4d
    LR: 0.1
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/resnet.py#L842
  In Collection: ResNeXt
- Name: resnext50d_32x4d
  Metadata:
    FLOPs: 5781119488
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - Grouped Convolution
    - Max Pooling
    - ReLU
    - ResNeXt Block
    - Residual Connection
    - Softmax
    File Size: 100515304
    Tasks:
    - Image Classification
    ID: resnext50d_32x4d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnet.py#L869
  In Collection: ResNeXt
Collections:
- Name: ResNeXt
  Paper:
    title: Aggregated Residual Transformations for Deep Neural Networks
    url: https://papperswithcode.com//paper/aggregated-residual-transformations-for-deep
  type: model-index
Type: model-index
-->