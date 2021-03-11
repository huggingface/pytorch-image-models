# Summary

Extending  “shallow” skip connections, **Dense Layer Aggregation (DLA)** incorporates more depth and sharing. The authors introduce two structures for deep layer aggregation (DLA): iterative deep aggregation (IDA) and hierarchical deep aggregation (HDA). These structures are expressed through an architectural framework, independent of the choice of backbone, for compatibility with current and future networks. 

IDA focuses on fusing resolutions and scales while HDA focuses on merging features from all modules and channels. IDA follows the base hierarchy to refine resolution and aggregate scale stage-bystage. HDA assembles its own hierarchy of tree-structured connections that cross and merge stages to aggregate different levels of representation. 

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('dla60', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `dla60`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('dla60', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{yu2019deep,
      title={Deep Layer Aggregation}, 
      author={Fisher Yu and Dequan Wang and Evan Shelhamer and Trevor Darrell},
      year={2019},
      eprint={1707.06484},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: dla60
  Metadata:
    FLOPs: 4256251880
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 89560235
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60
    LR: 0.1
    Layers: 60
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L394
  Config: ''
  In Collection: DLA
- Name: dla46_c
  Metadata:
    FLOPs: 583277288
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 5307963
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla46_c
    LR: 0.1
    Layers: 46
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L369
  Config: ''
  In Collection: DLA
- Name: dla102x2
  Metadata:
    FLOPs: 9343847400
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 167645295
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla102x2
    LR: 0.1
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L426
  Config: ''
  In Collection: DLA
- Name: dla102
  Metadata:
    FLOPs: 7192952808
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 135290579
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla102
    LR: 0.1
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L410
  Config: ''
  In Collection: DLA
- Name: dla102x
  Metadata:
    FLOPs: 5886821352
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 107552695
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla102x
    LR: 0.1
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L418
  Config: ''
  In Collection: DLA
- Name: dla169
  Metadata:
    FLOPs: 11598004200
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 8x GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 216547113
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla169
    LR: 0.1
    Layers: 169
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L434
  Config: ''
  In Collection: DLA
- Name: dla46x_c
  Metadata:
    FLOPs: 544052200
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 4387641
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla46x_c
    LR: 0.1
    Layers: 46
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L378
  Config: ''
  In Collection: DLA
- Name: dla60_res2net
  Metadata:
    FLOPs: 4147578504
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 84886593
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60_res2net
    Layers: 60
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L346
  Config: ''
  In Collection: DLA
- Name: dla60_res2next
  Metadata:
    FLOPs: 3485335272
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 69639245
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60_res2next
    Layers: 60
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L354
  Config: ''
  In Collection: DLA
- Name: dla34
  Metadata:
    FLOPs: 3070105576
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 63228658
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla34
    LR: 0.1
    Layers: 32
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L362
  Config: ''
  In Collection: DLA
- Name: dla60x
  Metadata:
    FLOPs: 3544204264
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 70883139
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60x
    LR: 0.1
    Layers: 60
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L402
  Config: ''
  In Collection: DLA
- Name: dla60x_c
  Metadata:
    FLOPs: 593325032
    Epochs: 120
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Convolution
    - DLA Bottleneck Residual Block
    - DLA Residual Block
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    File Size: 5454396
    Tasks:
    - Image Classification
    Training Time: ''
    ID: dla60x_c
    LR: 0.1
    Layers: 60
    Crop Pct: '0.875'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L386
  Config: ''
  In Collection: DLA
Collections:
- Name: DLA
  Paper:
    title: Deep Layer Aggregation
    url: https://papperswithcode.com//paper/deep-layer-aggregation
  type: model-index
Type: model-index
-->