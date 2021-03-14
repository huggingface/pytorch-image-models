# ECA-ResNet

An **ECA ResNet** is a variant on a [ResNet](https://paperswithcode.com/method/resnet) that utilises an [Efficient Channel Attention module](https://paperswithcode.com/method/efficient-channel-attention). Efficient Channel Attention is an architectural unit based on [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block) that reduces model complexity without dimensionality reduction. 

{% include 'code_snippets.md' %}

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
Type: model-index
Collections:
- Name: ECAResNet
  Paper:
    Title: 'ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks'
    URL: https://paperswithcode.com/paper/eca-net-efficient-channel-attention-for-deep
Models:
- Name: ecaresnet101d
  In Collection: ECAResNet
  Metadata:
    FLOPs: 10377193728
    Parameters: 44570000
    File Size: 178815067
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x RTX 2080Ti GPUs
    ID: ecaresnet101d
    LR: 0.1
    Epochs: 100
    Layers: 101
    Crop Pct: '0.875'
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1087
  Weights: https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNet101D_281c5844.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 82.18%
      Top 5 Accuracy: 96.06%
- Name: ecaresnet101d_pruned
  In Collection: ECAResNet
  Metadata:
    FLOPs: 4463972081
    Parameters: 24880000
    File Size: 99852736
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: ecaresnet101d_pruned
    Layers: 101
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1097
  Weights: https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45610/outputs/ECAResNet101D_P_75a3370e.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.82%
      Top 5 Accuracy: 95.64%
- Name: ecaresnet50d
  In Collection: ECAResNet
  Metadata:
    FLOPs: 5591090432
    Parameters: 25580000
    File Size: 102579290
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: 4x RTX 2080Ti GPUs
    ID: ecaresnet50d
    LR: 0.1
    Epochs: 100
    Layers: 50
    Crop Pct: '0.875'
    Batch Size: 256
    Image Size: '224'
    Weight Decay: 0.0001
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1045
  Weights: https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNet50D_833caf58.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.61%
      Top 5 Accuracy: 95.31%
- Name: ecaresnet50d_pruned
  In Collection: ECAResNet
  Metadata:
    FLOPs: 3250730657
    Parameters: 19940000
    File Size: 79990436
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: ecaresnet50d_pruned
    Layers: 50
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1055
  Weights: https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45899/outputs/ECAResNet50D_P_9c67f710.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 79.71%
      Top 5 Accuracy: 94.88%
- Name: ecaresnetlight
  In Collection: ECAResNet
  Metadata:
    FLOPs: 5276118784
    Parameters: 30160000
    File Size: 120956612
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
    Tasks:
    - Image Classification
    Training Techniques:
    - SGD with Momentum
    - Weight Decay
    Training Data:
    - ImageNet
    ID: ecaresnetlight
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/resnet.py#L1077
  Weights: https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNetLight_4f34b35b.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.46%
      Top 5 Accuracy: 95.25%
-->
