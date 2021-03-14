# (Gluon) Xception

**Xception** is a convolutional neural network architecture that relies solely on [depthwise separable convolution](https://paperswithcode.com/method/depthwise-separable-convolution) layers.

The weights from this model were ported from [Gluon](https://cv.gluon.ai/model_zoo/classification.html).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('gluon_xception65', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `gluon_xception65`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('gluon_xception65', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{chollet2017xception,
      title={Xception: Deep Learning with Depthwise Separable Convolutions}, 
      author={François Chollet},
      year={2017},
      eprint={1610.02357},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: Gloun Xception
  Paper:
    Title: 'Xception: Deep Learning with Depthwise Separable Convolutions'
    URL: https://paperswithcode.com/paper/xception-deep-learning-with-depthwise
Models:
- Name: gluon_xception65
  In Collection: Gloun Xception
  Metadata:
    FLOPs: 17594889728
    Parameters: 39920000
    File Size: 160551306
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: gluon_xception65
    Crop Pct: '0.903'
    Image Size: '299'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/gluon_xception.py#L241
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_xception-7015a15c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.7%
      Top 5 Accuracy: 94.87%
-->