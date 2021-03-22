# Feature Extraction

All of the models in `timm` have consistent mechanisms for obtaining various types of features from the model for tasks besides classification.

## Penultimate Layer Features (Pre-Classifier Features)

The features from the penultimate model layer can be obtained in several ways without requiring model surgery (although feel free to do surgery). One must first decide if they want pooled or un-pooled features.

### Unpooled

There are three ways to obtain unpooled features.

Without modifying the network, one can call `model.forward_features(input)` on any model instead of the usual `model(input)`. This will bypass the head classifier and global pooling for networks.

If one wants to explicitly modify the network to return unpooled features, they can either create the model without a classifier and pooling, or remove it later. Both paths remove the parameters associated with the classifier from the network.

#### forward_features()
```python hl_lines="3 6"
import torch
import timm
m = timm.create_model('xception41', pretrained=True)
o = m(torch.randn(2, 3, 299, 299))
print(f'Original shape: {o.shape}')
o = m.forward_features(torch.randn(2, 3, 299, 299))
print(f'Unpooled shape: {o.shape}')
```
Output:
```text
Original shape: torch.Size([2, 1000])
Unpooled shape: torch.Size([2, 2048, 10, 10])
```

#### Create with no classifier and pooling
```python hl_lines="3"
import torch
import timm
m = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='')
o = m(torch.randn(2, 3, 224, 224))
print(f'Unpooled shape: {o.shape}')
```
Output:
```text
Unpooled shape: torch.Size([2, 2048, 7, 7])
```

#### Remove it later
```python hl_lines="3 6"
import torch
import timm
m = timm.create_model('densenet121', pretrained=True)
o = m(torch.randn(2, 3, 224, 224))
print(f'Original shape: {o.shape}')
m.reset_classifier(0, '')
o = m(torch.randn(2, 3, 224, 224))
print(f'Unpooled shape: {o.shape}')
```
Output:
```text
Original shape: torch.Size([2, 1000])
Unpooled shape: torch.Size([2, 1024, 7, 7])
```

### Pooled

To modify the network to return pooled features, one can use `forward_features()` and pool/flatten the result themselves, or modify the network like above but keep pooling intact. 

#### Create with no classifier
```python hl_lines="3"
import torch
import timm
m = timm.create_model('resnet50', pretrained=True, num_classes=0)
o = m(torch.randn(2, 3, 224, 224))
print(f'Pooled shape: {o.shape}')
```
Output:
```text
Pooled shape: torch.Size([2, 2048])
```

#### Remove it later
```python hl_lines="3 6"
import torch
import timm
m = timm.create_model('ese_vovnet19b_dw', pretrained=True)
o = m(torch.randn(2, 3, 224, 224))
print(f'Original shape: {o.shape}')
m.reset_classifier(0)
o = m(torch.randn(2, 3, 224, 224))
print(f'Pooled shape: {o.shape}')
```
Output:
```text
Pooled shape: torch.Size([2, 1024])
```


## Multi-scale Feature Maps (Feature Pyramid)

Object detection, segmentation, keypoint, and a variety of dense pixel tasks require access to feature maps from the backbone network at multiple scales. This is often done by modifying the original classification network. Since each network varies quite a bit in structure, it's not uncommon to see only a few backbones supported in any given obj detection or segmentation library.

`timm` allows a consistent interface for creating any of the included models as feature backbones that output feature maps for selected levels. 

A feature backbone can be created by adding the argument `features_only=True` to any `create_model` call. By default 5 strides will be output from most models (not all have that many), with the first starting at 2 (some start at 1 or 4).

### Create a feature map extraction model
```python hl_lines="3"
import torch
import timm
m = timm.create_model('resnest26d', features_only=True, pretrained=True)
o = m(torch.randn(2, 3, 224, 224))
for x in o:
  print(x.shape)
```
Output:
```text
torch.Size([2, 64, 112, 112])
torch.Size([2, 256, 56, 56])
torch.Size([2, 512, 28, 28])
torch.Size([2, 1024, 14, 14])
torch.Size([2, 2048, 7, 7])
```

### Query the feature information

After a feature backbone has been created, it can be queried to provide channel or resolution reduction information to the downstream heads without requiring static config or hardcoded constants. The `.feature_info` attribute is a class encapsulating the information about the feature extraction points.

```python hl_lines="3 4"
import torch
import timm
m = timm.create_model('regnety_032', features_only=True, pretrained=True)
print(f'Feature channels: {m.feature_info.channels()}')
o = m(torch.randn(2, 3, 224, 224))
for x in o:
  print(x.shape)
```
Output:
```text
Feature channels: [32, 72, 216, 576, 1512]
torch.Size([2, 32, 112, 112])
torch.Size([2, 72, 56, 56])
torch.Size([2, 216, 28, 28])
torch.Size([2, 576, 14, 14])
torch.Size([2, 1512, 7, 7])
```

### Select specific feature levels or limit the stride

There are to additional creation arguments impacting the output features. 

* `out_indices` selects which indices to output
* `output_stride` limits the feature output stride of the network (also works in classification mode BTW)

`out_indices` is supported by all models, but not all models have the same index to feature stride mapping. Look at the code or check feature_info to compare. The out indices generally correspond to the `C(i+1)th` feature level (a `2^(i+1)` reduction). For most models, index 0 is the stride 2 features, and index 4 is stride 32.

`output_stride` is achieved by converting layers to use dilated convolutions. Doing so is not always straightforward, some networks only support `output_stride=32`.

```python hl_lines="3 4 5"
import torch
import timm
m = timm.create_model('ecaresnet101d', features_only=True, output_stride=8, out_indices=(2, 4), pretrained=True)
print(f'Feature channels: {m.feature_info.channels()}')
print(f'Feature reduction: {m.feature_info.reduction()}')
o = m(torch.randn(2, 3, 320, 320))
for x in o:
  print(x.shape)
```
Output:
```text
Feature channels: [512, 2048]
Feature reduction: [8, 8]
torch.Size([2, 512, 40, 40])
torch.Size([2, 2048, 40, 40])
```
