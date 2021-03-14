# SPNASNet

**Single-Path NAS** is a novel differentiable NAS method for designing hardware-efficient ConvNets in less than 4 hours.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('spnasnet_100', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `spnasnet_100`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('spnasnet_100', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{stamoulis2019singlepath,
      title={Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours}, 
      author={Dimitrios Stamoulis and Ruizhou Ding and Di Wang and Dimitrios Lymberopoulos and Bodhi Priyantha and Jie Liu and Diana Marculescu},
      year={2019},
      eprint={1904.02877},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!--
Type: model-index
Collections:
- Name: SPNASNet
  Paper:
    Title: 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4
      Hours'
    URL: https://paperswithcode.com/paper/single-path-nas-designing-hardware-efficient
Models:
- Name: spnasnet_100
  In Collection: SPNASNet
  Metadata:
    FLOPs: 442385600
    Parameters: 4420000
    File Size: 17902337
    Architecture:
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - ReLU
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: spnasnet_100
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L995
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.08%
      Top 5 Accuracy: 91.82%
-->