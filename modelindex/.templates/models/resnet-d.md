# Summary

**ResNet-D** is a modification on the [ResNet](https://paperswithcode.com/method/resnet) architecture that utilises an [average pooling](https://paperswithcode.com/method/average-pooling) tweak for downsampling. The motivation is that in the unmodified ResNet, the [1×1 convolution](https://paperswithcode.com/method/1x1-convolution) for the downsampling block ignores 3/4 of input feature maps, so this is modified so no information will be ignored

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{he2018bag,
      title={Bag of Tricks for Image Classification with Convolutional Neural Networks}, 
      author={Tong He and Zhi Zhang and Hang Zhang and Zhongyue Zhang and Junyuan Xie and Mu Li},
      year={2018},
      eprint={1812.01187},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Models:
- Name: resnet50d
  Metadata:
    FLOPs: 5591002624
    Training Data:
    - ImageNet
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
    File Size: 102567109
    Tasks:
    - Image Classification
    ID: resnet50d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L699
  In Collection: ResNet-D
- Name: resnet26d
  Metadata:
    FLOPs: 3335276032
    Training Data:
    - ImageNet
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
    File Size: 64209122
    Tasks:
    - Image Classification
    ID: resnet26d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L683
  In Collection: ResNet-D
- Name: resnet18d
  Metadata:
    FLOPs: 2645205760
    Training Data:
    - ImageNet
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
    File Size: 46893231
    Tasks:
    - Image Classification
    ID: resnet18d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L649
  In Collection: ResNet-D
- Name: resnet34d
  Metadata:
    FLOPs: 5026601728
    Training Data:
    - ImageNet
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
    File Size: 87369807
    Tasks:
    - Image Classification
    ID: resnet34d
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L666
  In Collection: ResNet-D
- Name: resnet200d
  Metadata:
    FLOPs: 26034378752
    Training Data:
    - ImageNet
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
    File Size: 259662933
    Tasks:
    - Image Classification
    ID: resnet200d
    Crop Pct: '0.94'
    Image Size: '256'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L749
  In Collection: ResNet-D
- Name: resnet101d
  Metadata:
    FLOPs: 13805639680
    Training Data:
    - ImageNet
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
    File Size: 178791263
    Tasks:
    - Image Classification
    ID: resnet101d
    Crop Pct: '0.94'
    Image Size: '256'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L716
  In Collection: ResNet-D
- Name: resnet152d
  Metadata:
    FLOPs: 20155275264
    Training Data:
    - ImageNet
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
    File Size: 241596837
    Tasks:
    - Image Classification
    ID: resnet152d
    Crop Pct: '0.94'
    Image Size: '256'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/resnet.py#L724
  In Collection: ResNet-D
Collections:
- Name: ResNet-D
  Paper:
    title: Bag of Tricks for Image Classification with Convolutional Neural Networks
    url: https://paperswithcode.com//paper/bag-of-tricks-for-image-classification-with
  type: model-index
Type: model-index
-->
