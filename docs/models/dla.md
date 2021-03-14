# Deep Layer Aggregation

Extending  “shallow” skip connections, **Dense Layer Aggregation (DLA)** incorporates more depth and sharing. The authors introduce two structures for deep layer aggregation (DLA): iterative deep aggregation (IDA) and hierarchical deep aggregation (HDA). These structures are expressed through an architectural framework, independent of the choice of backbone, for compatibility with current and future networks. 

IDA focuses on fusing resolutions and scales while HDA focuses on merging features from all modules and channels. IDA follows the base hierarchy to refine resolution and aggregate scale stage-bystage. HDA assembles its own hierarchy of tree-structured connections that cross and merge stages to aggregate different levels of representation. 

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('dla102', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `dla102`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('dla102', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
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
Type: model-index
Collections:
- Name: DLA
  Paper:
    Title: Deep Layer Aggregation
    URL: https://paperswithcode.com/paper/deep-layer-aggregation
Models:
- Name: dla102
  In Collection: DLA
  Metadata:
    FLOPs: 7192952808
    Parameters: 33270000
    File Size: 135290579
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    ID: dla102
    LR: 0.1
    Epochs: 120
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L410
  Weights: http://dl.yf.io/dla/models/imagenet/dla102-d94d9790.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.03%
      Top 5 Accuracy: 93.95%
- Name: dla102x
  In Collection: DLA
  Metadata:
    FLOPs: 5886821352
    Parameters: 26310000
    File Size: 107552695
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    ID: dla102x
    LR: 0.1
    Epochs: 120
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L418
  Weights: http://dl.yf.io/dla/models/imagenet/dla102x-ad62be81.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.51%
      Top 5 Accuracy: 94.23%
- Name: dla102x2
  In Collection: DLA
  Metadata:
    FLOPs: 9343847400
    Parameters: 41280000
    File Size: 167645295
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    ID: dla102x2
    LR: 0.1
    Epochs: 120
    Layers: 102
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L426
  Weights: http://dl.yf.io/dla/models/imagenet/dla102x2-262837b6.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.44%
      Top 5 Accuracy: 94.65%
- Name: dla169
  In Collection: DLA
  Metadata:
    FLOPs: 11598004200
    Parameters: 53390000
    File Size: 216547113
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 8x GPUs
    ID: dla169
    LR: 0.1
    Epochs: 120
    Layers: 169
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L434
  Weights: http://dl.yf.io/dla/models/imagenet/dla169-0914e092.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.69%
      Top 5 Accuracy: 94.33%
- Name: dla34
  In Collection: DLA
  Metadata:
    FLOPs: 3070105576
    Parameters: 15740000
    File Size: 63228658
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: dla34
    LR: 0.1
    Epochs: 120
    Layers: 32
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L362
  Weights: http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 74.62%
      Top 5 Accuracy: 92.06%
- Name: dla46_c
  In Collection: DLA
  Metadata:
    FLOPs: 583277288
    Parameters: 1300000
    File Size: 5307963
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: dla46_c
    LR: 0.1
    Epochs: 120
    Layers: 46
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L369
  Weights: http://dl.yf.io/dla/models/imagenet/dla46_c-2bfd52c3.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 64.87%
      Top 5 Accuracy: 86.29%
- Name: dla46x_c
  In Collection: DLA
  Metadata:
    FLOPs: 544052200
    Parameters: 1070000
    File Size: 4387641
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: dla46x_c
    LR: 0.1
    Epochs: 120
    Layers: 46
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L378
  Weights: http://dl.yf.io/dla/models/imagenet/dla46x_c-d761bae7.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 65.98%
      Top 5 Accuracy: 86.99%
- Name: dla60
  In Collection: DLA
  Metadata:
    FLOPs: 4256251880
    Parameters: 22040000
    File Size: 89560235
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: dla60
    LR: 0.1
    Epochs: 120
    Layers: 60
    Dropout: 0.2
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L394
  Weights: http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.04%
      Top 5 Accuracy: 93.32%
- Name: dla60_res2net
  In Collection: DLA
  Metadata:
    FLOPs: 4147578504
    Parameters: 20850000
    File Size: 84886593
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: dla60_res2net
    Layers: 60
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L346
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net_dla60_4s-d88db7f9.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.46%
      Top 5 Accuracy: 94.21%
- Name: dla60_res2next
  In Collection: DLA
  Metadata:
    FLOPs: 3485335272
    Parameters: 17030000
    File Size: 69639245
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: dla60_res2next
    Layers: 60
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L354
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2next_dla60_4s-d327927b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.44%
      Top 5 Accuracy: 94.16%
- Name: dla60x
  In Collection: DLA
  Metadata:
    FLOPs: 3544204264
    Parameters: 17350000
    File Size: 70883139
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: dla60x
    LR: 0.1
    Epochs: 120
    Layers: 60
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L402
  Weights: http://dl.yf.io/dla/models/imagenet/dla60x-d15cacda.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.25%
      Top 5 Accuracy: 94.02%
- Name: dla60x_c
  In Collection: DLA
  Metadata:
    FLOPs: 593325032
    Parameters: 1320000
    File Size: 5454396
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: dla60x_c
    LR: 0.1
    Epochs: 120
    Layers: 60
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bilinear
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dla.py#L386
  Weights: http://dl.yf.io/dla/models/imagenet/dla60x_c-b870c45c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 67.91%
      Top 5 Accuracy: 88.42%
-->