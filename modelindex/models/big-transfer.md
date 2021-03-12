# Summary

**Big Transfer (BiT)** is a type of pretraining recipe that pre-trains  on a large supervised source dataset, and fine-tunes the weights on the target task. Models are trained on the JFT-300M dataset. The finetuned models contained in this collection are finetuned on ImageNet.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('resnetv2_152x4_bitm', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `resnetv2_152x4_bitm`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('resnetv2_152x4_bitm', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{kolesnikov2020big,
      title={Big Transfer (BiT): General Visual Representation Learning}, 
      author={Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Joan Puigcerver and Jessica Yung and Sylvain Gelly and Neil Houlsby},
      year={2020},
      eprint={1912.11370},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: resnetv2_152x4_bitm
  Metadata:
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 3746270104
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_152x4_bitm
    Crop Pct: '1.0'
    Image Size: '480'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L465
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_152x2_bitm
  Metadata:
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 945476668
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_152x2_bitm
    Crop Pct: '1.0'
    Image Size: '480'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L458
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_50x1_bitm
  Metadata:
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 102242668
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_50x1_bitm
    LR: 0.03
    Layers: 50
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L430
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_101x3_bitm
  Metadata:
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 1551830100
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_101x3_bitm
    LR: 0.03
    Layers: 101
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L451
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_50x3_bitm
  Metadata:
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 869321580
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_50x3_bitm
    LR: 0.03
    Layers: 50
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L437
  Config: ''
  In Collection: Big Transfer
- Name: resnetv2_101x1_bitm
  Metadata:
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Mixup
    - SGD with Momentum
    - Weight Decay
    Training Resources: Cloud TPUv3-512
    Architecture:
    - 1x1 Convolution
    - Bottleneck Residual Block
    - Convolution
    - Global Average Pooling
    - Group Normalization
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Weight Standardization
    File Size: 178256468
    Tasks:
    - Image Classification
    Training Time: ''
    ID: resnetv2_101x1_bitm
    LR: 0.03
    Layers: 101
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '480'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/resnetv2.py#L444
  Config: ''
  In Collection: Big Transfer
Collections:
- Name: Big Transfer
  Paper:
    title: 'Big Transfer (BiT): General Visual Representation Learning'
    url: https://paperswithcode.com//paper/large-scale-learning-of-general-visual
  type: model-index
Type: model-index
-->