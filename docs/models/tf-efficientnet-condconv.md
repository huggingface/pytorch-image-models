# (Tensorflow) EfficientNet CondConv

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a *compound coefficient*. Unlike conventional practice that arbitrary scales  these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2^N$ times more computational resources, then we can simply increase the network depth by $\alpha ^ N$,  width by $\beta ^ N$, and image size by $\gamma ^ N$, where $\alpha, \beta, \gamma$ are constant coefficients determined by a small grid search on the original small model. EfficientNet uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a  principled way.

The compound scaling method is justified by the intuition that if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2), in addition to squeeze-and-excitation blocks.

This collection of models amends EfficientNet by adding [CondConv](https://paperswithcode.com/method/condconv) convolutions.

The weights from this model were ported from [Tensorflow/TPU](https://github.com/tensorflow/tpu).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('tf_efficientnet_cc_b0_4e', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `tf_efficientnet_cc_b0_4e`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('tf_efficientnet_cc_b0_4e', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/abs-1904-04971,
  author    = {Brandon Yang and
               Gabriel Bender and
               Quoc V. Le and
               Jiquan Ngiam},
  title     = {Soft Conditional Computation},
  journal   = {CoRR},
  volume    = {abs/1904.04971},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.04971},
  archivePrefix = {arXiv},
  eprint    = {1904.04971},
  timestamp = {Thu, 25 Apr 2019 13:55:01 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1904-04971.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<!--
Type: model-index
Collections:
- Name: TF EfficientNet CondConv
  Paper:
    Title: 'CondConv: Conditionally Parameterized Convolutions for Efficient Inference'
    URL: https://paperswithcode.com/paper/soft-conditional-computation
Models:
- Name: tf_efficientnet_cc_b0_4e
  In Collection: TF EfficientNet CondConv
  Metadata:
    FLOPs: 224153788
    Parameters: 13310000
    File Size: 53490940
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - CondConv
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_cc_b0_4e
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1561
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_4e-4362b6b2.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.32%
      Top 5 Accuracy: 93.32%
- Name: tf_efficientnet_cc_b0_8e
  In Collection: TF EfficientNet CondConv
  Metadata:
    FLOPs: 224158524
    Parameters: 24010000
    File Size: 96287616
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - CondConv
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_cc_b0_8e
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1572
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_8e-66184a25.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.91%
      Top 5 Accuracy: 93.65%
- Name: tf_efficientnet_cc_b1_8e
  In Collection: TF EfficientNet CondConv
  Metadata:
    FLOPs: 370427824
    Parameters: 39720000
    File Size: 159206198
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - CondConv
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_cc_b1_8e
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.882'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '240'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1584
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b1_8e-f7c79ae1.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.33%
      Top 5 Accuracy: 94.37%
-->