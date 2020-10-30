# Getting Started

## Install

The library can be installed with pip:

```
pip install timm
```

!!! info "Conda Environment"
    All development and testing has been done in Conda Python 3 environments on Linux x86-64 systems, specifically Python 3.6.x, 3.7.x., 3.8.x.
    
    Little to no care has been taken to be Python 2.x friendly and will not support it. If you run into any challenges running on Windows, or other OS, I'm definitely open to looking into those issues so long as it's in a reproducible (read Conda) environment.
    
    PyTorch versions 1.4, 1.5.x, 1.6, and 1.7 have been tested with this code.
    
    I've tried to keep the dependencies minimal, the setup is as per the PyTorch default install instructions for Conda:
    ```
    conda create -n torch-env
    conda activate torch-env
    conda install -c pytorch pytorch torchvision cudatoolkit=11
    conda install pyyaml
    ```

## Load a Pretrained Model

Pretrained models can be loaded using `timm.create_model`

```python
import timm

m = timm.create_model('mobilenetv3_large_100', pretrained=True)
m.eval()
```

## List Models with Pretrained Weights
```python
import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)
>>> ['adv_inception_v3',
 'cspdarknet53',
 'cspresnext50',
 'densenet121',
 'densenet161',
 'densenet169',
 'densenet201',
 'densenetblur121d',
 'dla34',
 'dla46_c',
...
]
```

## List Model Architectures by Wildcard
```python
import timm
from pprint import pprint
model_names = timm.list_models('*resne*t*')
pprint(model_names)
>>> ['cspresnet50',
 'cspresnet50d',
 'cspresnet50w',
 'cspresnext50',
...
]
```
