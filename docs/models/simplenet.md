# SimpleNet v1

**SimpleNetV1** is a convolutional neural network that is designed with simplicity in mind and outperforms much deeper and more complex architectures. The network design includes the most basic operators in a plain CNN network and is completely comprised of just Convolutional layers followed by BatchNormalization and ReLU activation function.  

## How do I use this model on an image?
To load a pretrained model:
simply choose a variant among available ones and do as follows:
```python
import timm
model = timm.create_model('simplenetv1_5m_m2', pretrained=True)
model.eval()
```

To load and preprocess the image:
```python 
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension
```

To get the model predictions:
```python
import torch
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
# prints: torch.Size([1000])
```

To get the top-5 predictions class names:
```python
# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
# prints class names and probabilities like:
# Samoyed 0.8442415595054626
# Pomeranian 0.09159677475690842
# Great Pyrenees 0.013929242268204689
# Arctic fox 0.00913404393941164
# white wolf 0.008576451800763607
```

Replace the model name with the variant you want to use, e.g. `simplenetv1_9m_m2`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.

## How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('simplenetv1_5m_m2', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.

## How do I train this model?

You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@article{DBLP:journals/corr/HasanPourRVS16,
  author    = {Seyyed Hossein HasanPour and
               Mohammad Rouhani and
               Mohsen Fayyaz and
               Mohammad Sabokrou},
  title     = {Lets keep it simple, Using simple architectures to outperform deeper
               and more complex architectures},
  journal   = {CoRR},
  volume    = {abs/1608.06037},
  year      = {2016},
  url       = {http://arxiv.org/abs/1608.06037},
  eprinttype = {arXiv},
  eprint    = {1608.06037},
  timestamp = {Mon, 13 Aug 2018 16:48:25 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HasanPourRVS16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
