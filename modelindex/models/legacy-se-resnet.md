# Summary

**SE ResNet** is a variant of a [ResNet](https://www.paperswithcode.com/method/resnet) that employs [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) to enable the network to perform dynamic channel-wise feature recalibration.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('legacy_seresnet101', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `legacy_seresnet101`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('legacy_seresnet101', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
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
Models:
- Name: legacy_seresnet101
  Metadata:
    FLOPs: 9762614000
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA Titan X GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 197822624
    Tasks:
    - Image Classification
    ID: legacy_seresnet101
    LR: 0.6
    Layers: 101
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L426
  In Collection: Legacy SE ResNet
- Name: legacy_seresnet152
  Metadata:
    FLOPs: 14553578160
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA Titan X GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 268033864
    Tasks:
    - Image Classification
    ID: legacy_seresnet152
    LR: 0.6
    Layers: 152
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L433
  In Collection: Legacy SE ResNet
- Name: legacy_seresnet18
  Metadata:
    FLOPs: 2328876024
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA Titan X GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 47175663
    Tasks:
    - Image Classification
    ID: legacy_seresnet18
    LR: 0.6
    Layers: 18
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L405
  In Collection: Legacy SE ResNet
- Name: legacy_seresnet34
  Metadata:
    FLOPs: 4706201004
    Epochs: 100
    Batch Size: 1024
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA Titan X GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 87958697
    Tasks:
    - Image Classification
    ID: legacy_seresnet34
    LR: 0.6
    Layers: 34
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L412
  In Collection: Legacy SE ResNet
- Name: legacy_seresnet50
  Metadata:
    FLOPs: 4974351024
    Epochs: 100
    Training Data:
    - ImageNet
    Training Techniques:
    - Label Smoothing
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x NVIDIA Titan X GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 112611220
    Tasks:
    - Image Classification
    ID: legacy_seresnet50
    LR: 0.6
    Layers: 50
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Interpolation: bilinear
    Minibatch Size: 1024
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/senet.py#L419
  In Collection: Legacy SE ResNet
Collections:
- Name: Legacy SE ResNet
  Paper:
    title: Squeeze-and-Excitation Networks
    url: https://papperswithcode.com//paper/squeeze-and-excitation-networks
  type: model-index
Type: model-index
-->