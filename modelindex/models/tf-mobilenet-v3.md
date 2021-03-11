# Summary

**MobileNetV3** is a convolutional neural network that is designed for mobile phone CPUs. The network design includes the use of a [hard swish activation](https://paperswithcode.com/method/hard-swish) and [squeeze-and-excitation](https://paperswithcode.com/method/squeeze-and-excitation-block) modules in the [MBConv blocks](https://paperswithcode.com/method/inverted-residual-block).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('tf_mobilenetv3_large_075', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `tf_mobilenetv3_large_075`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('tf_mobilenetv3_large_075', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/abs-1905-02244,
  author    = {Andrew Howard and
               Mark Sandler and
               Grace Chu and
               Liang{-}Chieh Chen and
               Bo Chen and
               Mingxing Tan and
               Weijun Wang and
               Yukun Zhu and
               Ruoming Pang and
               Vijay Vasudevan and
               Quoc V. Le and
               Hartwig Adam},
  title     = {Searching for MobileNetV3},
  journal   = {CoRR},
  volume    = {abs/1905.02244},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.02244},
  archivePrefix = {arXiv},
  eprint    = {1905.02244},
  timestamp = {Tue, 12 Jan 2021 15:30:06 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1905-02244.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Models:
- Name: tf_mobilenetv3_large_075
  Metadata:
    FLOPs: 194323712
    Batch Size: 4096
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 4x4 TPU Pod
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Hard Swish
    - Inverted Residual Block
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 16097377
    Tasks:
    - Image Classification
    ID: tf_mobilenetv3_large_075
    LR: 0.1
    Dropout: 0.8
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L394
  In Collection: TF MobileNet V3
- Name: tf_mobilenetv3_large_100
  Metadata:
    FLOPs: 274535288
    Batch Size: 4096
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 4x4 TPU Pod
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Hard Swish
    - Inverted Residual Block
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 22076649
    Tasks:
    - Image Classification
    ID: tf_mobilenetv3_large_100
    LR: 0.1
    Dropout: 0.8
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L403
  In Collection: TF MobileNet V3
- Name: tf_mobilenetv3_large_minimal_100
  Metadata:
    FLOPs: 267216928
    Batch Size: 4096
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 4x4 TPU Pod
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Hard Swish
    - Inverted Residual Block
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 15836368
    Tasks:
    - Image Classification
    ID: tf_mobilenetv3_large_minimal_100
    LR: 0.1
    Dropout: 0.8
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L412
  In Collection: TF MobileNet V3
- Name: tf_mobilenetv3_small_075
  Metadata:
    FLOPs: 48457664
    Batch Size: 4096
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 16x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Hard Swish
    - Inverted Residual Block
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 8242701
    Tasks:
    - Image Classification
    ID: tf_mobilenetv3_small_075
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bilinear
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L421
  In Collection: TF MobileNet V3
- Name: tf_mobilenetv3_small_100
  Metadata:
    FLOPs: 65450600
    Batch Size: 4096
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 16x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Hard Swish
    - Inverted Residual Block
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 10256398
    Tasks:
    - Image Classification
    ID: tf_mobilenetv3_small_100
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bilinear
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L430
  In Collection: TF MobileNet V3
- Name: tf_mobilenetv3_small_minimal_100
  Metadata:
    FLOPs: 60827936
    Batch Size: 4096
    Training Data:
    - ImageNet
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Resources: 16x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Hard Swish
    - Inverted Residual Block
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 8258083
    Tasks:
    - Image Classification
    ID: tf_mobilenetv3_small_minimal_100
    LR: 0.045
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 4.0e-05
    Interpolation: bilinear
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/mobilenetv3.py#L439
  In Collection: TF MobileNet V3
Collections:
- Name: TF MobileNet V3
  Paper:
    title: Searching for MobileNetV3
    url: https://papperswithcode.com//paper/searching-for-mobilenetv3
  type: model-index
Type: model-index
-->