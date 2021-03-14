# EfficientNet (Knapsack Pruned)

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a *compound coefficient*. Unlike conventional practice that arbitrary scales  these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2^N$ times more computational resources, then we can simply increase the network depth by $\alpha ^ N$,  width by $\beta ^ N$, and image size by $\gamma ^ N$, where $\alpha, \beta, \gamma$ are constant coefficients determined by a small grid search on the original small model. EfficientNet uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a  principled way.

The compound scaling method is justified by the intuition that if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2), in addition to [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block).

This collection consists of pruned EfficientNet models.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{tan2020efficientnet,
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2020},
      eprint={1905.11946},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{aflalo2020knapsack,
      title={Knapsack Pruning with Inner Distillation},
      author={Yonathan Aflalo and Asaf Noy and Ming Lin and Itamar Friedman and Lihi Zelnik},
      year={2020},
      eprint={2002.08258},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!--
Type: model-index
Collections:
- Name: EfficientNet Pruned
  Paper:
    Title: Knapsack Pruning with Inner Distillation
    URL: https://paperswithcode.com/paper/knapsack-pruning-with-inner-distillation
Models:
- Name: efficientnet_b1_pruned
  In Collection: EfficientNet Pruned
  Metadata:
    FLOPs: 489653114
    Parameters: 6330000
    File Size: 25595162
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b1_pruned
    Crop Pct: '0.882'
    Image Size: '240'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1208
  Weights: https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb1_pruned_9ebb3fe6.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.25%
      Top 5 Accuracy: 93.84%
- Name: efficientnet_b2_pruned
  In Collection: EfficientNet Pruned
  Metadata:
    FLOPs: 878133915
    Parameters: 8310000
    File Size: 33555005
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b2_pruned
    Crop Pct: '0.89'
    Image Size: '260'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1219
  Weights: https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb2_pruned_203f55bc.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.91%
      Top 5 Accuracy: 94.86%
- Name: efficientnet_b3_pruned
  In Collection: EfficientNet Pruned
  Metadata:
    FLOPs: 1239590641
    Parameters: 9860000
    File Size: 39770812
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: efficientnet_b3_pruned
    Crop Pct: '0.904'
    Image Size: '300'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/efficientnet.py#L1230
  Weights: https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb3_pruned_5abcc29f.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.86%
      Top 5 Accuracy: 95.24%
-->
