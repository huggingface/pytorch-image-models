# Getting Started


## Install

The library can be installed with pip:

```
pip install timm
```

!!! info "Conda Environment"
    All development and testing has been done in Conda Python 3 environments
     on Linux x86-64 systems, specifically Python 3.6.x and 3.7.x. 

    To install `timm` in a conda environment:
    ```
    conda create -n torch-env
    conda activate torch-env
    conda install -c pytorch pytorch torchvision cudatoolkit=10.1
    conda install pyyaml
    pip install timm
    ```


## Load Pretrained Model

Pretrained models can be loaded using `timm.create_model`

```python
import timm

m = timm.create_model('mobilenetv3_100', pretrained=True)
m.eval()
```

To load a different model see [the list of pretrained weights](/models
/#pretrained-imagenet-weights).