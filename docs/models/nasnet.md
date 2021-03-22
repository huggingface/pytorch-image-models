# NASNet

**NASNet** is a type of convolutional neural network discovered through neural architecture search. The building blocks consist of normal and reduction cells.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('nasnetalarge', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `nasnetalarge`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('nasnetalarge', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{zoph2018learning,
      title={Learning Transferable Architectures for Scalable Image Recognition}, 
      author={Barret Zoph and Vijay Vasudevan and Jonathon Shlens and Quoc V. Le},
      year={2018},
      eprint={1707.07012},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: NASNet
  Paper:
    Title: Learning Transferable Architectures for Scalable Image Recognition
    URL: https://paperswithcode.com/paper/learning-transferable-architectures-for
Models:
- Name: nasnetalarge
  In Collection: NASNet
  Metadata:
    FLOPs: 30242402862
    Parameters: 88750000
    File Size: 356056626
    Architecture:
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - ReLU
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 50x Tesla K40 GPUs
    ID: nasnetalarge
    Dropout: 0.5
    Crop Pct: '0.911'
    Momentum: 0.9
    Image Size: '331'
    Interpolation: bicubic
    Label Smoothing: 0.1
    RMSProp $\epsilon$: 1.0
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/nasnet.py#L562
  Weights: http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.63%
      Top 5 Accuracy: 96.05%
-->