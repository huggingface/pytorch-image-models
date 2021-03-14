# SelecSLS

**SelecSLS** uses novel selective long and short range skip connections to improve the information flow allowing for a drastically faster network without compromising accuracy.

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{Mehta_2020,
   title={XNect},
   volume={39},
   ISSN={1557-7368},
   url={http://dx.doi.org/10.1145/3386569.3392410},
   DOI={10.1145/3386569.3392410},
   number={4},
   journal={ACM Transactions on Graphics},
   publisher={Association for Computing Machinery (ACM)},
   author={Mehta, Dushyant and Sotnychenko, Oleksandr and Mueller, Franziska and Xu, Weipeng and Elgharib, Mohamed and Fua, Pascal and Seidel, Hans-Peter and Rhodin, Helge and Pons-Moll, Gerard and Theobalt, Christian},
   year={2020},
   month={Jul}
}
```

<!--
Type: model-index
Collections:
- Name: SelecSLS
  Paper:
    Title: 'XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera'
    URL: https://paperswithcode.com/paper/xnect-real-time-multi-person-3d-human-pose
Models:
- Name: selecsls42b
  In Collection: SelecSLS
  Metadata:
    FLOPs: 3824022528
    Parameters: 32460000
    File Size: 129948954
    Architecture:
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - ReLU
    - SelecSLS Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Random Erasing
    Training Data:
    - ImageNet
    ID: selecsls42b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/selecsls.py#L335
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-selecsls/selecsls42b-8af30141.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.18%
      Top 5 Accuracy: 93.39%
- Name: selecsls60
  In Collection: SelecSLS
  Metadata:
    FLOPs: 4610472600
    Parameters: 30670000
    File Size: 122839714
    Architecture:
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - ReLU
    - SelecSLS Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Random Erasing
    Training Data:
    - ImageNet
    ID: selecsls60
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/selecsls.py#L342
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-selecsls/selecsls60-bbf87526.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.99%
      Top 5 Accuracy: 93.83%
- Name: selecsls60b
  In Collection: SelecSLS
  Metadata:
    FLOPs: 4657653144
    Parameters: 32770000
    File Size: 131252898
    Architecture:
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - ReLU
    - SelecSLS Block
    Tasks:
    - Image Classification
    Training Techniques:
    - Cosine Annealing
    - Random Erasing
    Training Data:
    - ImageNet
    ID: selecsls60b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/selecsls.py#L349
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-selecsls/selecsls60b-94e619b5.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.41%
      Top 5 Accuracy: 94.18%
-->
