# (Tensorflow) EfficientNet Lite

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a *compound coefficient*. Unlike conventional practice that arbitrary scales  these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2^N$ times more computational resources, then we can simply increase the network depth by $\alpha ^ N$,  width by $\beta ^ N$, and image size by $\gamma ^ N$, where $\alpha, \beta, \gamma$ are constant coefficients determined by a small grid search on the original small model. EfficientNet uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a  principled way.

The compound scaling method is justified by the intuition that if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2).

EfficientNet-Lite makes EfficientNet more suitable for mobile devices by introducing [ReLU6](https://paperswithcode.com/method/relu6) activation functions and removing [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation).

The weights from this model were ported from [Tensorflow/TPU](https://github.com/tensorflow/tpu).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `tf_efficientnet_lite0`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('tf_efficientnet_lite0', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{tan2020efficientnet,
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2020},
      eprint={1905.11946},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!--
Type: model-index
Collections:
- Name: TF EfficientNet Lite
  Paper:
    Title: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
    URL: https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for
Models:
- Name: tf_efficientnet_lite0
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 488052032
    Parameters: 4650000
    File Size: 18820223
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite0
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1596
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.83%
      Top 5 Accuracy: 92.17%
- Name: tf_efficientnet_lite1
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 773639520
    Parameters: 5420000
    File Size: 21939331
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite1
    Crop Pct: '0.882'
    Image Size: '240'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1607
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.67%
      Top 5 Accuracy: 93.24%
- Name: tf_efficientnet_lite2
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 1068494432
    Parameters: 6090000
    File Size: 24658687
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite2
    Crop Pct: '0.89'
    Image Size: '260'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1618
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite2-dcccb7df.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.48%
      Top 5 Accuracy: 93.75%
- Name: tf_efficientnet_lite3
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 2011534304
    Parameters: 8199999
    File Size: 33161413
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite3
    Crop Pct: '0.904'
    Image Size: '300'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1629
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.83%
      Top 5 Accuracy: 94.91%
- Name: tf_efficientnet_lite4
  In Collection: TF EfficientNet Lite
  Metadata:
    FLOPs: 5164802912
    Parameters: 13010000
    File Size: 52558819
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - RELU6
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_lite4
    Crop Pct: '0.92'
    Image Size: '380'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1640
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite4-741542c3.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.54%
      Top 5 Accuracy: 95.66%
-->