# Vision Transformer (ViT)

The **Vision Transformer** is a model for image classification that employs a Transformer-like architecture over patches of the image. This includes the use of [Multi-Head Attention](https://paperswithcode.com/method/multi-head-attention), [Scaled Dot-Product Attention](https://paperswithcode.com/method/scaled) and other architectural features seen in the [Transformer](https://paperswithcode.com/method/transformer) architecture traditionally used for NLP.

{% include 'code_snippets.md' %}

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
Models:
- Name: vit_large_patch16_384
  Metadata:
    FLOPs: 174702764032
    Batch Size: 512
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Resources: TPUv3
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
    File Size: 1218907013
    Tasks:
    - Image Classification
    Training Time: ''
    ID: vit_large_patch16_384
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '384'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L561
  Config: ''
  In Collection: Vision Transformer
- Name: vit_base_patch16_224
  Metadata:
    FLOPs: 67394605056
    Epochs: 90
    Batch Size: 4096
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Resources: TPUv3
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
    File Size: 346292833
    Tasks:
    - Image Classification
    Training Time: ''
    ID: vit_base_patch16_224
    LR: 0.0008
    Dropout: 0.0
    Crop Pct: '0.9'
    Image Size: '224'
    Warmup Steps: 10000
    Weight Decay: 0.03
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L503
  Config: ''
  In Collection: Vision Transformer
- Name: vit_base_patch16_384
  Metadata:
    FLOPs: 49348245504
    Batch Size: 512
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Resources: TPUv3
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
    File Size: 347460194
    Tasks:
    - Image Classification
    Training Time: ''
    ID: vit_base_patch16_384
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '384'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L522
  Config: ''
  In Collection: Vision Transformer
- Name: vit_large_patch16_224
  Metadata:
    FLOPs: 119294746624
    Batch Size: 512
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Resources: TPUv3
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
    File Size: 1217350532
    Tasks:
    - Image Classification
    Training Time: ''
    ID: vit_large_patch16_224
    Crop Pct: '0.9'
    Momentum: 0.9
    Image Size: '224'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L542
  Config: ''
  In Collection: Vision Transformer
- Name: vit_base_patch32_384
  Metadata:
    FLOPs: 12656142336
    Batch Size: 512
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Resources: TPUv3
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
    File Size: 353210979
    Tasks:
    - Image Classification
    Training Time: ''
    ID: vit_base_patch32_384
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '384'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L532
  Config: ''
  In Collection: Vision Transformer
- Name: vit_base_resnet50_384
  Metadata:
    FLOPs: 49461491712
    Batch Size: 512
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Resources: TPUv3
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
    File Size: 395854632
    Tasks:
    - Image Classification
    Training Time: ''
    ID: vit_base_resnet50_384
    Crop Pct: '1.0'
    Momentum: 0.9
    Image Size: '384'
    Weight Decay: 0.0
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L653
  Config: ''
  In Collection: Vision Transformer
- Name: vit_small_patch16_224
  Metadata:
    FLOPs: 28236450816
    Training Data:
    - ImageNet
    - JFT-300M
    Training Techniques:
    - Cosine Annealing
    - Gradient Clipping
    - SGD with Momentum
    Training Resources: TPUv3
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
    File Size: 195031454
    Tasks:
    - Image Classification
    Training Time: ''
    ID: vit_small_patch16_224
    Crop Pct: '0.9'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L490
  Config: ''
  In Collection: Vision Transformer
Collections:
- Name: Vision Transformer
  Paper:
    title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
    url: https://paperswithcode.com//paper/an-image-is-worth-16x16-words-transformers-1
  type: model-index
Type: model-index
-->
