# Summary

**Inception v3** is a convolutional neural network architecture from the Inception family that makes several improvements including using [Label Smoothing](https://paperswithcode.com/method/label-smoothing), Factorized 7 x 7 convolutions, and the use of an [auxiliary classifer](https://paperswithcode.com/method/auxiliary-classifier) to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead). The key building block is an [Inception Module](https://paperswithcode.com/method/inception-v3-module).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('inception_v3', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `inception_v3`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('inception_v3', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/SzegedyVISW15,
  author    = {Christian Szegedy and
               Vincent Vanhoucke and
               Sergey Ioffe and
               Jonathon Shlens and
               Zbigniew Wojna},
  title     = {Rethinking the Inception Architecture for Computer Vision},
  journal   = {CoRR},
  volume    = {abs/1512.00567},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.00567},
  archivePrefix = {arXiv},
  eprint    = {1512.00567},
  timestamp = {Mon, 13 Aug 2018 16:49:07 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/SzegedyVISW15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Models:
- Name: inception_v3
  Metadata:
    FLOPs: 7352418880
    Training Data:
    - ImageNet
    Training Techniques:
    - Gradient Clipping
    - Label Smoothing
    - RMSProp
    - Weight Decay
    Training Resources: 50x NVIDIA Kepler GPUs
    Architecture:
    - 1x1 Convolution
    - Auxiliary Classifier
    - Average Pooling
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inception-v3 Module
    - Max Pooling
    - ReLU
    - Softmax
    File Size: 108857766
    Tasks:
    - Image Classification
    ID: inception_v3
    LR: 0.045
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/inception_v3.py#L442
  In Collection: Inception v3
Collections:
- Name: Inception v3
  Paper:
    title: Rethinking the Inception Architecture for Computer Vision
    url: https://paperswithcode.com//paper/rethinking-the-inception-architecture-for
  type: model-index
Type: model-index
-->