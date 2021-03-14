# MnasNet

**MnasNet** is a type of convolutional neural network optimized for mobile devices that is discovered through mobile neural architecture search, which explicitly incorporates model latency into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. The main building block is an [inverted residual block](https://paperswithcode.com/method/inverted-residual-block) (from [MobileNetV2](https://paperswithcode.com/method/mobilenetv2)).

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('mnasnet_100', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `mnasnet_100`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('mnasnet_100', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{tan2019mnasnet,
      title={MnasNet: Platform-Aware Neural Architecture Search for Mobile}, 
      author={Mingxing Tan and Bo Chen and Ruoming Pang and Vijay Vasudevan and Mark Sandler and Andrew Howard and Quoc V. Le},
      year={2019},
      eprint={1807.11626},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: MNASNet
  Paper:
    Title: 'MnasNet: Platform-Aware Neural Architecture Search for Mobile'
    URL: https://paperswithcode.com/paper/mnasnet-platform-aware-neural-architecture
Models:
- Name: mnasnet_100
  In Collection: MNASNet
  Metadata:
    FLOPs: 416415488
    Parameters: 4380000
    File Size: 17731774
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Inverted Residual Block
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - RMSProp
    - Weight Decay
    Training Data:
    - ImageNet
    ID: mnasnet_100
    Layers: 100
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 4000
    Image Size: '224'
    Interpolation: bicubic
    RMSProp Decay: 0.9
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L894
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.67%
      Top 5 Accuracy: 92.1%
- Name: semnasnet_100
  In Collection: MNASNet
  Metadata:
    FLOPs: 414570766
    Parameters: 3890000
    File Size: 15731489
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - Depthwise Separable Convolution
    - Dropout
    - Global Average Pooling
    - Inverted Residual Block
    - Max Pooling
    - ReLU
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: semnasnet_100
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L928
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 75.45%
      Top 5 Accuracy: 92.61%
-->