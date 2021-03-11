# Summary

An **ECA ResNet** is a variant on a [ResNet](https://paperswithcode.com/method/resnet) that utilises an [Efficient Channel Attention module](https://paperswithcode.com/method/efficient-channel-attention). Efficient Channel Attention is an architectural unit based on [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) that reduces model complexity without dimensionality reduction. 

## How do I use this model on an image?
To load a pretrained model:

```python
import timm
model = timm.create_model('ecaresnet101d', pretrained=True)
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

Replace the model name with the variant you want to use, e.g. `ecaresnet101d`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('ecaresnet101d', pretrained=True).reset_classifier(NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{wang2020ecanet,
      title={ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks}, 
      author={Qilong Wang and Banggu Wu and Pengfei Zhu and Peihua Li and Wangmeng Zuo and Qinghua Hu},
      year={2020},
      eprint={1910.03151},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: ecaresnet101d
  Metadata:
    FLOPs: 10377193728
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 4x RTX 2080Ti GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 178815067
    Tasks:
    - Image Classification
    ID: ecaresnet101d
    LR: 0.1
    Layers: 101
    Crop Pct: '0.875'
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1087
  In Collection: ECAResNet
- Name: ecaresnet101d_pruned
  Metadata:
    FLOPs: 4463972081
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: ''
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 99852736
    Tasks:
    - Image Classification
    Training Time: ''
    ID: ecaresnet101d_pruned
    Layers: 101
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1097
  Config: ''
  In Collection: ECAResNet
- Name: ecaresnet50d_pruned
  Metadata:
    FLOPs: 3250730657
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 79990436
    Tasks:
    - Image Classification
    ID: ecaresnet50d_pruned
    Layers: 50
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1055
  In Collection: ECAResNet
- Name: ecaresnet50d
  Metadata:
    FLOPs: 5591090432
    Epochs: 100
    Batch Size: 256
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 4x RTX 2080Ti GPUs
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 102579290
    Tasks:
    - Image Classification
    ID: ecaresnet50d
    LR: 0.1
    Layers: 50
    Crop Pct: '0.875'
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1045
  In Collection: ECAResNet
- Name: ecaresnetlight
  Metadata:
    FLOPs: 5276118784
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Architecture:
    - 1x1 Convolution
    - Batch Normalization
    - Bottleneck Residual Block
    - Convolution
    - Efficient Channel Attention
    - Global Average Pooling
    - Max Pooling
    - ReLU
    - Residual Block
    - Residual Connection
    - Softmax
    - Squeeze-and-Excitation Block
    File Size: 120956612
    Tasks:
    - Image Classification
    ID: ecaresnetlight
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1077
  In Collection: ECAResNet
Collections:
- Name: ECAResNet
  Paper:
    title: 'ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks'
    url: https://papperswithcode.com//paper/eca-net-efficient-channel-attention-for-deep
  type: model-index
Type: model-index
-->