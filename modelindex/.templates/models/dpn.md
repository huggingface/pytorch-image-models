# Summary

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
Models:
- Name: dpn68
  Metadata:
    FLOPs: 2990567880
    Batch Size: 1280
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 40x K80 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    File Size: 50761994
    Tasks:
    - Image Classification
    ID: dpn68
    LR: 0.316
    Layers: 68
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L270
  In Collection: DPN
- Name: dpn68b
  Metadata:
    FLOPs: 2990567880
    Batch Size: 1280
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 40x K80 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    File Size: 50781025
    Tasks:
    - Image Classification
    ID: dpn68b
    LR: 0.316
    Layers: 68
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L278
  In Collection: DPN
- Name: dpn92
  Metadata:
    FLOPs: 8357659624
    Batch Size: 1280
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 40x K80 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    File Size: 151248422
    Tasks:
    - Image Classification
    ID: dpn92
    LR: 0.316
    Layers: 92
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L286
  In Collection: DPN
- Name: dpn131
  Metadata:
    FLOPs: 20586274792
    Batch Size: 960
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 40x K80 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    File Size: 318016207
    Tasks:
    - Image Classification
    ID: dpn131
    LR: 0.316
    Layers: 131
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L302
  In Collection: DPN
- Name: dpn107
  Metadata:
    FLOPs: 23524280296
    Batch Size: 1280
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 40x K80 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    File Size: 348612331
    Tasks:
    - Image Classification
    ID: dpn107
    LR: 0.316
    Layers: 107
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L310
  In Collection: DPN
- Name: dpn98
  Metadata:
    FLOPs: 15003675112
    Batch Size: 1280
    Training Data:
    - ImageNet
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Resources: 40x K80 GPUs
    Architecture:
    - Batch Normalization
    - Convolution
    - DPN Block
    - Dense Connections
    - Global Average Pooling
    - Max Pooling
    - Softmax
    File Size: 247021307
    Tasks:
    - Image Classification
    ID: dpn98
    LR: 0.4
    Layers: 98
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/d8e69206be253892b2956341fea09fdebfaae4e3/timm/models/dpn.py#L294
  In Collection: DPN
Collections:
- Name: DPN
  Paper:
    title: Dual Path Networks
    url: https://paperswithcode.com//paper/dual-path-networks
  type: model-index
Type: model-index
-->
