# Res2Net

**Res2Net** is an image model that employs a variation on bottleneck residual blocks, [Res2Net Blocks](https://paperswithcode.com/method/res2net-block). The motivation is to be able to represent features at multiple scales. This is achieved through a novel building block for CNNs that constructs hierarchical residual-like connections within one single residual block. This represents multi-scale features at a granular level and increases the range of receptive fields for each network layer.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('res2net101_26w_4s', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `res2net101_26w_4s`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('res2net101_26w_4s', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{Gao_2021,
   title={Res2Net: A New Multi-Scale Backbone Architecture},
   volume={43},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/TPAMI.2019.2938758},
   DOI={10.1109/tpami.2019.2938758},
   number={2},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
   year={2021},
   month={Feb},
   pages={652–662}
}
```

<!--
Type: model-index
Collections:
- Name: Res2Net
  Paper:
    Title: 'Res2Net: A New Multi-scale Backbone Architecture'
    URL: https://paperswithcode.com/paper/res2net-a-new-multi-scale-backbone
Models:
- Name: res2net101_26w_4s
  In Collection: Res2Net
  Metadata:
    FLOPs: 10415881200
    Parameters: 45210000
    File Size: 181456059
    Architecture:
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - ReLU
    - Res2Net Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x Titan Xp GPUs
    ID: res2net101_26w_4s
    LR: 0.1
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/res2net.py#L152
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net101_26w_4s-02a759a1.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.19%
      Top 5 Accuracy: 94.43%
- Name: res2net50_14w_8s
  In Collection: Res2Net
  Metadata:
    FLOPs: 5403546768
    Parameters: 25060000
    File Size: 100638543
    Architecture:
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - ReLU
    - Res2Net Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x Titan Xp GPUs
    ID: res2net50_14w_8s
    LR: 0.1
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/res2net.py#L196
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_14w_8s-6527dddc.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.14%
      Top 5 Accuracy: 93.86%
- Name: res2net50_26w_4s
  In Collection: Res2Net
  Metadata:
    FLOPs: 5499974064
    Parameters: 25700000
    File Size: 103110087
    Architecture:
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - ReLU
    - Res2Net Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x Titan Xp GPUs
    ID: res2net50_26w_4s
    LR: 0.1
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/res2net.py#L141
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_4s-06e79181.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.99%
      Top 5 Accuracy: 93.85%
- Name: res2net50_26w_6s
  In Collection: Res2Net
  Metadata:
    FLOPs: 8130156528
    Parameters: 37050000
    File Size: 148603239
    Architecture:
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - ReLU
    - Res2Net Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x Titan Xp GPUs
    ID: res2net50_26w_6s
    LR: 0.1
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/res2net.py#L163
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_6s-19041792.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.57%
      Top 5 Accuracy: 94.12%
- Name: res2net50_26w_8s
  In Collection: Res2Net
  Metadata:
    FLOPs: 10760338992
    Parameters: 48400000
    File Size: 194085165
    Architecture:
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - ReLU
    - Res2Net Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x Titan Xp GPUs
    ID: res2net50_26w_8s
    LR: 0.1
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/res2net.py#L174
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_8s-2c7c9f12.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.19%
      Top 5 Accuracy: 94.37%
- Name: res2net50_48w_2s
  In Collection: Res2Net
  Metadata:
    FLOPs: 5375291520
    Parameters: 25290000
    File Size: 101421406
    Architecture:
    - Batch Normalization
    - Convolution
    - Global Average Pooling
    - ReLU
    - Res2Net Block
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x Titan Xp GPUs
    ID: res2net50_48w_2s
    LR: 0.1
    Epochs: 100
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/res2net.py#L185
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_48w_2s-afed724a.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.53%
      Top 5 Accuracy: 93.56%
-->