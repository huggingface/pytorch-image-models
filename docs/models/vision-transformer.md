# Vision Transformer (ViT)

The **Vision Transformer** is a model for image classification that employs a Transformer-like architecture over patches of the image. This includes the use of [Multi-Head Attention](https://paperswithcode.com/method/multi-head-attention), [Scaled Dot-Product Attention](https://paperswithcode.com/method/scaled) and other architectural features seen in the [Transformer](https://paperswithcode.com/method/transformer) architecture traditionally used for NLP.

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `vit_base_patch16_224`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{dosovitskiy2020image,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2020},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: Vision Transformer
  Paper:
    Title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
    URL: https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1
Models:
- Name: vit_base_patch16_224
  In Collection: Vision Transformer
  Metadata:
    FLOPs: 67394605056
    Parameters: 86570000
    File Size: 346292833
    Architecture:
    - Attention Dropout
    - Convolution
    - Dense Connections
    - Dropout
    - GELU
    - Layer Normalization
    - Multi-Head Attention
    - Scaled Dot-Product Attention
    - Tanh Activation
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: TPUv3
    ID: vit_base_patch16_224
    LR: 0.0008
    Epochs: 90
    Dropout: 0.0
    Crop Pct: '0.9'
    Batch Size: 4096
    Image Size: '224'
    Warmup Steps: 10000
    Weight Decay: 0.03
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L503
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.78%
      Top 5 Accuracy: 96.13%
- Name: vit_base_patch16_384
  In Collection: Vision Transformer
  Metadata:
    FLOPs: 49348245504
    Parameters: 86860000
    File Size: 347460194
    Architecture:
    - Attention Dropout
    - Convolution
    - Dense Connections
    - Dropout
    - GELU
    - Layer Normalization
    - Multi-Head Attention
    - Scaled Dot-Product Attention
    - Tanh Activation
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: TPUv3
    ID: vit_base_patch16_384
    Crop Pct: '1.0'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '384'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L522
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 84.2%
      Top 5 Accuracy: 97.22%
- Name: vit_base_patch32_384
  In Collection: Vision Transformer
  Metadata:
    FLOPs: 12656142336
    Parameters: 88300000
    File Size: 353210979
    Architecture:
    - Attention Dropout
    - Convolution
    - Dense Connections
    - Dropout
    - GELU
    - Layer Normalization
    - Multi-Head Attention
    - Scaled Dot-Product Attention
    - Tanh Activation
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: TPUv3
    ID: vit_base_patch32_384
    Crop Pct: '1.0'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '384'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L532
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.66%
      Top 5 Accuracy: 96.13%
- Name: vit_base_resnet50_384
  In Collection: Vision Transformer
  Metadata:
    FLOPs: 49461491712
    Parameters: 98950000
    File Size: 395854632
    Architecture:
    - Attention Dropout
    - Convolution
    - Dense Connections
    - Dropout
    - GELU
    - Layer Normalization
    - Multi-Head Attention
    - Scaled Dot-Product Attention
    - Tanh Activation
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: TPUv3
    ID: vit_base_resnet50_384
    Crop Pct: '1.0'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '384'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L653
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 84.99%
      Top 5 Accuracy: 97.3%
- Name: vit_large_patch16_224
  In Collection: Vision Transformer
  Metadata:
    FLOPs: 119294746624
    Parameters: 304330000
    File Size: 1217350532
    Architecture:
    - Attention Dropout
    - Convolution
    - Dense Connections
    - Dropout
    - GELU
    - Layer Normalization
    - Multi-Head Attention
    - Scaled Dot-Product Attention
    - Tanh Activation
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: TPUv3
    ID: vit_large_patch16_224
    Crop Pct: '0.9'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '224'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L542
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 83.06%
      Top 5 Accuracy: 96.44%
- Name: vit_large_patch16_384
  In Collection: Vision Transformer
  Metadata:
    FLOPs: 174702764032
    Parameters: 304720000
    File Size: 1218907013
    Architecture:
    - Attention Dropout
    - Convolution
    - Dense Connections
    - Dropout
    - GELU
    - Layer Normalization
    - Multi-Head Attention
    - Scaled Dot-Product Attention
    - Tanh Activation
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: TPUv3
    ID: vit_large_patch16_384
    Crop Pct: '1.0'
    Momentum: 0.9
    Batch Size: 512
    Image Size: '384'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L561
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 85.17%
      Top 5 Accuracy: 97.36%
- Name: vit_small_patch16_224
  In Collection: Vision Transformer
  Metadata:
    FLOPs: 28236450816
    Parameters: 48750000
    File Size: 195031454
    Architecture:
    - Attention Dropout
    - Convolution
    - Dense Connections
    - Dropout
    - GELU
    - Layer Normalization
    - Multi-Head Attention
    - Scaled Dot-Product Attention
    - Tanh Activation
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: TPUv3
    ID: vit_small_patch16_224
    Crop Pct: '0.9'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L490
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.85%
      Top 5 Accuracy: 93.42%
-->