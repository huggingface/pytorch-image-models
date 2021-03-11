# Summary

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
Models:
- Name: selecsls42b
  Metadata:
    FLOPs: 3824022528
    Training Data:
    - ImageNet
    Training Techniques:
    - Cosine Annealing
    - Random Erasing
    Architecture:
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - ReLU
    - SelecSLS Block
    File Size: 129948954
    Tasks:
    - Image Classification
    ID: selecsls42b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/selecsls.py#L335
  In Collection: SelecSLS
- Name: selecsls60
  Metadata:
    FLOPs: 4610472600
    Training Data:
    - ImageNet
    Training Techniques:
    - Cosine Annealing
    - Random Erasing
    Architecture:
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - ReLU
    - SelecSLS Block
    File Size: 122839714
    Tasks:
    - Image Classification
    ID: selecsls60
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/selecsls.py#L342
  In Collection: SelecSLS
- Name: selecsls60b
  Metadata:
    FLOPs: 4657653144
    Training Data:
    - ImageNet
    Training Techniques:
    - Cosine Annealing
    - Random Erasing
    Architecture:
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Global Average Pooling
    - ReLU
    - SelecSLS Block
    File Size: 131252898
    Tasks:
    - Image Classification
    ID: selecsls60b
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/b9843f954b0457af2db4f9dea41a8538f51f5d78/timm/models/selecsls.py#L349
  In Collection: SelecSLS
Collections:
- Name: SelecSLS
  Paper:
    title: 'XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera'
    url: https://papperswithcode.com//paper/xnect-real-time-multi-person-3d-human-pose
  type: model-index
Type: model-index
-->
