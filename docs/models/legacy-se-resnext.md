# (Legacy) SE-ResNeXt

**SE ResNeXt** is a variant of a [ResNeXt](https://www.paperswithcode.com/method/resnext) that employs [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) to enable the network to perform dynamic channel-wise feature recalibration.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('legacy_seresnext101_32x4d', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `legacy_seresnext101_32x4d`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('legacy_seresnext101_32x4d', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{hu2019squeezeandexcitation,
      title={Squeeze-and-Excitation Networks}, 
      author={Jie Hu and Li Shen and Samuel Albanie and Gang Sun and Enhua Wu},
      year={2019},
      eprint={1709.01507},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: Legacy SE ResNeXt
  Paper:
    Title: Squeeze-and-Excitation Networks
    URL: https://paperswithcode.com/paper/squeeze-and-excitation-networks
Models:
- Name: legacy_seresnext101_32x4d
  In Collection: Legacy SE ResNeXt
  Metadata:
    FLOPs: 10287698672
    Parameters: 48960000
    File Size: 196466866
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
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: legacy_seresnext101_32x4d
    LR: 0.6
    Epochs: 100
    Layers: 101
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L462
  Weights: http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.23%
      Top 5 Accuracy: 95.02%
- Name: legacy_seresnext26_32x4d
  In Collection: Legacy SE ResNeXt
  Metadata:
    FLOPs: 3187342304
    Parameters: 16790000
    File Size: 67346327
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
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: legacy_seresnext26_32x4d
    LR: 0.6
    Epochs: 100
    Layers: 26
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L448
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26_32x4d-65ebdb501.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.11%
      Top 5 Accuracy: 93.31%
- Name: legacy_seresnext50_32x4d
  In Collection: Legacy SE ResNeXt
  Metadata:
    FLOPs: 5459954352
    Parameters: 27560000
    File Size: 110559176
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
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x NVIDIA Titan X GPUs
    ID: legacy_seresnext50_32x4d
    LR: 0.6
    Epochs: 100
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 1024
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L455
  Weights: http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.08%
      Top 5 Accuracy: 94.43%
-->