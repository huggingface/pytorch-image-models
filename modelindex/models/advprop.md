# Summary

**AdvProp** is an adversarial training scheme which treats adversarial examples as additional examples, to prevent overfitting. Key to the method is the usage of a separate auxiliary batch norm for adversarial examples, as they have different underlying distributions to normal examples.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('tf_efficientnet_b1_ap', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `tf_efficientnet_b1_ap`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('tf_efficientnet_b1_ap', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{xie2020adversarial,
      title={Adversarial Examples Improve Image Recognition}, 
      author={Cihang Xie and Mingxing Tan and Boqing Gong and Jiang Wang and Alan Yuille and Quoc V. Le},
      year={2020},
      eprint={1911.09665},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: tf_efficientnet_b1_ap
  Metadata:
    FLOPs: 883633200
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 31515350
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b1_ap
    LR: 0.256
    Crop Pct: '0.882'
    Momentum: 0.9
    Image Size: '240'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1344
  In Collection: AdvProp
- Name: tf_efficientnet_b2_ap
  Metadata:
    FLOPs: 1234321170
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 36800745
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b2_ap
    LR: 0.256
    Crop Pct: '0.89'
    Momentum: 0.9
    Image Size: '260'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1354
  In Collection: AdvProp
- Name: tf_efficientnet_b3_ap
  Metadata:
    FLOPs: 2275247568
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 49384538
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b3_ap
    LR: 0.256
    Crop Pct: '0.904'
    Momentum: 0.9
    Image Size: '300'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1364
  In Collection: AdvProp
- Name: tf_efficientnet_b4_ap
  Metadata:
    FLOPs: 5749638672
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 77993585
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b4_ap
    LR: 0.256
    Crop Pct: '0.922'
    Momentum: 0.9
    Image Size: '380'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1374
  In Collection: AdvProp
- Name: tf_efficientnet_b5_ap
  Metadata:
    FLOPs: 13176501888
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 122403150
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b5_ap
    LR: 0.256
    Crop Pct: '0.934'
    Momentum: 0.9
    Image Size: '456'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1384
  In Collection: AdvProp
- Name: tf_efficientnet_b6_ap
  Metadata:
    FLOPs: 24180518488
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 173237466
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b6_ap
    LR: 0.256
    Crop Pct: '0.942'
    Momentum: 0.9
    Image Size: '528'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1394
  In Collection: AdvProp
- Name: tf_efficientnet_b7_ap
  Metadata:
    FLOPs: 48205304880
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 266850607
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b7_ap
    LR: 0.256
    Crop Pct: '0.949'
    Momentum: 0.9
    Image Size: '600'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1405
  In Collection: AdvProp
- Name: tf_efficientnet_b8_ap
  Metadata:
    FLOPs: 80962956270
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 351412563
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b8_ap
    LR: 0.128
    Crop Pct: '0.954'
    Momentum: 0.9
    Image Size: '672'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1416
  In Collection: AdvProp
- Name: tf_efficientnet_b0_ap
  Metadata:
    FLOPs: 488688572
    Epochs: 350
    Batch Size: 2048
    Training Data:
    - ImageNet
    Training Techniques:
    - AdvProp
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    File Size: 21385973
    Tasks:
    - Image Classification
    ID: tf_efficientnet_b0_ap
    LR: 0.256
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1334
  In Collection: AdvProp
Collections:
- Name: AdvProp
  Paper:
    title: Adversarial Examples Improve Image Recognition
    url: https://paperswithcode.com//paper/adversarial-examples-improve-image
  type: model-index
Type: model-index
-->