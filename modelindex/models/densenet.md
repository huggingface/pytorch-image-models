# Summary

**DenseNet** is a type of convolutional neural network that utilises dense connections between layers, through [Dense Blocks](http://www.paperswithcode.com/method/dense-block), where we connect *all layers* (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers.

The **DenseNet Blur** variant in this collection by Ross Wightman employs [Blur Pooling](http://www.paperswithcode.com/method/blur-pooling)

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('densenetblur121d', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `densenetblur121d`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('densenetblur121d', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/HuangLW16a,
  author    = {Gao Huang and
               Zhuang Liu and
               Kilian Q. Weinberger},
  title     = {Densely Connected Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1608.06993},
  year      = {2016},
  url       = {http://arxiv.org/abs/1608.06993},
  archivePrefix = {arXiv},
  eprint    = {1608.06993},
  timestamp = {Mon, 10 Sep 2018 15:49:32 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HuangLW16a.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

<!--
Models:
- Name: densenetblur121d
  Metadata:
    FLOPs: 3947812864
    Training Data:
    - ImageNet
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Blur Pooling
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    File Size: 32456500
    Tasks:
    - Image Classification
    ID: densenetblur121d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L305
  In Collection: DenseNet
- Name: tv_densenet121
  Metadata:
    FLOPs: 3641843200
    Epochs: 90
    Batch Size: 32
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    File Size: 32342954
    Tasks:
    - Image Classification
    ID: tv_densenet121
    LR: 0.1
    Crop Pct: '0.875'
    LR Gamma: 0.1
    Momentum: 0.9
    Image Size: '224'
    LR Step Size: 30
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L379
  In Collection: DenseNet
- Name: densenet121
  Metadata:
    FLOPs: 3641843200
    Epochs: 90
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Kaiming Initialization
    - Nesterov Accelerated Gradient
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    File Size: 32376726
    Tasks:
    - Image Classification
    Training Time: ''
    ID: densenet121
    LR: 0.1
    Layers: 121
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L295
  Config: ''
  In Collection: DenseNet
- Name: densenet201
  Metadata:
    FLOPs: 5514321024
    Epochs: 90
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Kaiming Initialization
    - Nesterov Accelerated Gradient
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    File Size: 81131730
    Tasks:
    - Image Classification
    ID: densenet201
    LR: 0.1
    Layers: 201
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L337
  In Collection: DenseNet
- Name: densenet169
  Metadata:
    FLOPs: 4316945792
    Epochs: 90
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Kaiming Initialization
    - Nesterov Accelerated Gradient
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    File Size: 57365526
    Tasks:
    - Image Classification
    ID: densenet169
    LR: 0.1
    Layers: 169
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L327
  In Collection: DenseNet
- Name: densenet161
  Metadata:
    FLOPs: 9931959264
    Epochs: 90
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - Kaiming Initialization
    - Nesterov Accelerated Gradient
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Block
    - Dense Connections
    - Dropout
    - Max Pooling
    - ReLU
    - Softmax
    File Size: 115730790
    Tasks:
    - Image Classification
    ID: densenet161
    LR: 0.1
    Layers: 161
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/densenet.py#L347
  In Collection: DenseNet
Collections:
- Name: DenseNet
  Paper:
    title: Densely Connected Convolutional Networks
    url: https://paperswithcode.com//paper/densely-connected-convolutional-networks
  type: model-index
Type: model-index
-->