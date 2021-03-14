# Summary

A **ResNest** is a variant on a [ResNet](https://paperswithcode.com/method/resnet), which instead stacks [Split-Attention blocks](https://paperswithcode.com/method/split-attention). The cardinal group representations are then concatenated along the channel dimension: $V = \text{Concat}${$V^{1},V^{2},\cdots{V}^{K}$}. As in standard residual blocks, the final output $Y$ of otheur Split-Attention block is produced using a shortcut connection: $Y=V+X$, if the input and output feature-map share the same shape.  For blocks with a stride, an appropriate transformation $\mathcal{T}$ is applied to the shortcut connection to align the output shapes:  $Y=V+\mathcal{T}(X)$. For example, $\mathcal{T}$ can be strided convolution or combined convolution-with-pooling.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('resnest50d_4s2x40d', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `resnest50d_4s2x40d`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('resnest50d_4s2x40d', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{zhang2020resnest,
      title={ResNeSt: Split-Attention Networks}, 
      author={Hang Zhang and Chongruo Wu and Zhongyue Zhang and Yi Zhu and Haibin Lin and Zhi Zhang and Yue Sun and Tong He and Jonas Mueller and R. Manmatha and Mu Li and Alexander Smola},
      year={2020},
      eprint={2004.08955},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: resnest50d_4s2x40d
  Metadata:
    FLOPs: 5657064720
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 122133282
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest50d_4s2x40d
    LR: 0.1
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L218
  Config: ''
  In Collection: ResNeSt
- Name: resnest200e
  Metadata:
    FLOPs: 45954387872
    Epochs: 270
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 193782911
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest200e
    LR: 0.1
    Layers: 200
    Dropout: 0.2
    Crop Pct: '0.909'
    Momentum: 0.9
    Image Size: '320'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L194
  Config: ''
  In Collection: ResNeSt
- Name: resnest14d
  Metadata:
    FLOPs: 3548594464
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 42562639
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest14d
    LR: 0.1
    Layers: 14
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L148
  Config: ''
  In Collection: ResNeSt
- Name: resnest101e
  Metadata:
    FLOPs: 17423183648
    Epochs: 270
    Batch Size: 4096
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 193782911
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest101e
    LR: 0.1
    Layers: 101
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '256'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L182
  Config: ''
  In Collection: ResNeSt
- Name: resnest269e
  Metadata:
    FLOPs: 100830307104
    Epochs: 270
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 445402691
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest269e
    LR: 0.1
    Layers: 269
    Dropout: 0.2
    Crop Pct: '0.928'
    Momentum: 0.9
    Image Size: '416'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L206
  Config: ''
  In Collection: ResNeSt
- Name: resnest26d
  Metadata:
    FLOPs: 4678918720
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 68470242
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest26d
    LR: 0.1
    Layers: 26
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L159
  Config: ''
  In Collection: ResNeSt
- Name: resnest50d
  Metadata:
    FLOPs: 6937106336
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 110273258
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnest50d
    LR: 0.1
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L170
  Config: ''
  In Collection: ResNeSt
- Name: resnest50d_1s4x24d
  Metadata:
    FLOPs: 5686764544
    Epochs: 270
    Batch Size: 8192
    Training Data:
    - ImageNet
    Training Techniques:
    - AutoAugment
    - DropBlock
    - Label Smoothing
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: 64x NVIDIA V100 GPUs
    Architecture:
    - 1x1 Convolution
    - Convolution
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Split Attention
    File Size: 103045531
    Tasks:
    - Image Classification
    ID: resnest50d_1s4x24d
    LR: 0.1
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnest.py#L229
  In Collection: ResNeSt
Collections:
- Name: ResNeSt
  Paper:
    title: 'ResNeSt: Split-Attention Networks'
    url: https://paperswithcode.com//paper/resnest-split-attention-networks
  type: model-index
Type: model-index
-->