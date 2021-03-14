# MixNet

**MixNet** is a type of convolutional neural network discovered via AutoML that utilises [MixConvs](https://paperswithcode.com/method/mixconv) instead of regular [depthwise convolutions](https://paperswithcode.com/method/depthwise-convolution).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('mixnet_l', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `mixnet_l`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('mixnet_l', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{tan2019mixconv,
      title={MixConv: Mixed Depthwise Convolutional Kernels}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2019},
      eprint={1907.09595},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: MixNet
  Paper:
    Title: 'MixConv: Mixed Depthwise Convolutional Kernels'
    URL: https://paperswithcode.com/paper/mixnet-mixed-depthwise-convolutional-kernels
Models:
- Name: mixnet_l
  In Collection: MixNet
  Metadata:
    FLOPs: 738671316
    Parameters: 7330000
    File Size: 29608232
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - MNAS
    Training Data:
    - ImageNet
    ID: mixnet_l
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1669
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.98%
      Top 5 Accuracy: 94.18%
- Name: mixnet_m
  In Collection: MixNet
  Metadata:
    FLOPs: 454543374
    Parameters: 5010000
    File Size: 20298347
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - MNAS
    Training Data:
    - ImageNet
    ID: mixnet_m
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1660
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.27%
      Top 5 Accuracy: 93.42%
- Name: mixnet_s
  In Collection: MixNet
  Metadata:
    FLOPs: 321264910
    Parameters: 4130000
    File Size: 16727982
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - MNAS
    Training Data:
    - ImageNet
    ID: mixnet_s
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1651
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.99%
      Top 5 Accuracy: 92.79%
- Name: mixnet_xl
  In Collection: MixNet
  Metadata:
    FLOPs: 1195880424
    Parameters: 11900000
    File Size: 48001170
    Architecture:
    - Batch Normalization
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - Grouped Convolution
    - MixConv
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - MNAS
    Training Data:
    - ImageNet
    ID: mixnet_xl
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1678
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.47%
      Top 5 Accuracy: 94.93%
-->