# Dual Path Network (DPN)

A **Dual Path Network (DPN)** is a convolutional neural network which presents a new topology of connection paths internally. The intuition is that [ResNets](https://paperswithcode.com/method/resnet) enables feature re-usage while DenseNet enables new feature exploration, and both are important for learning good representations. To enjoy the benefits from both path topologies, Dual Path Networks share common features while maintaining the flexibility to explore new features through dual path architectures. 

The principal building block is an [DPN Block](https://paperswithcode.com/method/dpn-block).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{chen2017dual,
      title={Dual Path Networks}, 
      author={Yunpeng Chen and Jianan Li and Huaxin Xiao and Xiaojie Jin and Shuicheng Yan and Jiashi Feng},
      year={2017},
      eprint={1707.01629},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!--
Type: model-index
Collections:
- Name: DPN
  Paper:
    Title: Dual Path Networks
    URL: https://paperswithcode.com/paper/dual-path-networks
Models:
- Name: dpn107
  In Collection: DPN
  Metadata:
    FLOPs: 23524280296
    Parameters: 86920000
    File Size: 348612331
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 40x K80 GPUs
    ID: dpn107
    LR: 0.316
    Layers: 107
    Crop Pct: '0.875'
    Batch Size: 1280
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L310
  Weights: https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn107_extra-1ac7121e2.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.16%
      Top 5 Accuracy: 94.91%
- Name: dpn131
  In Collection: DPN
  Metadata:
    FLOPs: 20586274792
    Parameters: 79250000
    File Size: 318016207
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 40x K80 GPUs
    ID: dpn131
    LR: 0.316
    Layers: 131
    Crop Pct: '0.875'
    Batch Size: 960
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L302
  Weights: https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn131-71dfe43e0.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.83%
      Top 5 Accuracy: 94.71%
- Name: dpn68
  In Collection: DPN
  Metadata:
    FLOPs: 2990567880
    Parameters: 12610000
    File Size: 50761994
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 40x K80 GPUs
    ID: dpn68
    LR: 0.316
    Layers: 68
    Crop Pct: '0.875'
    Batch Size: 1280
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L270
  Weights: https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68-66bebafa7.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.31%
      Top 5 Accuracy: 92.97%
- Name: dpn68b
  In Collection: DPN
  Metadata:
    FLOPs: 2990567880
    Parameters: 12610000
    File Size: 50781025
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 40x K80 GPUs
    ID: dpn68b
    LR: 0.316
    Layers: 68
    Crop Pct: '0.875'
    Batch Size: 1280
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L278
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dpn68b_ra-a31ca160.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.21%
      Top 5 Accuracy: 94.42%
- Name: dpn92
  In Collection: DPN
  Metadata:
    FLOPs: 8357659624
    Parameters: 37670000
    File Size: 151248422
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 40x K80 GPUs
    ID: dpn92
    LR: 0.316
    Layers: 92
    Crop Pct: '0.875'
    Batch Size: 1280
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L286
  Weights: https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn92_extra-b040e4a9b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.99%
      Top 5 Accuracy: 94.84%
- Name: dpn98
  In Collection: DPN
  Metadata:
    FLOPs: 15003675112
    Parameters: 61570000
    File Size: 247021307
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 40x K80 GPUs
    ID: dpn98
    LR: 0.4
    Layers: 98
    Crop Pct: '0.875'
    Batch Size: 1280
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L294
  Weights: https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn98-5b90dec4d.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.65%
      Top 5 Accuracy: 94.61%
-->
