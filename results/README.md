# Validation Results

This folder contains validation results for the models in this collection having pretrained weights. Since the focus for this repository is currently ImageNet-1k classification, all of the results are based on datasets compatible with ImageNet-1k classes.

## Datasets

There are currently results for the ImageNet validation set and 3 additional test sets.

### ImageNet Validation - [`results-imagenet.csv`](results-imagenet.csv)

* Source: http://image-net.org/challenges/LSVRC/2012/index
* Paper: "ImageNet Large Scale Visual Recognition Challenge" - https://arxiv.org/abs/1409.0575

The standard 50,000 image ImageNet-1k validation set. Model selection during training utilizes this validation set, so it is not a true test set. Question: Does anyone have the official ImageNet-1k test set classification labels now that challenges are done?

### ImageNetV2 Matched Frequency - [`results-imagenetv2-matched-frequency.csv`](results-imagenetv2-matched-frequency.csv)

* Source: https://github.com/modestyachts/ImageNetV2
* Paper: "Do ImageNet Classifiers Generalize to ImageNet?" - https://arxiv.org/abs/1902.10811

An ImageNet test set of 10,000 images sampled from new images roughly 10 years after the original. Care was taken to replicate the original ImageNet curation/sampling process.

### ImageNet-Sketch - [`results-sketch.csv`](results-sketch.csv)

* Source: https://github.com/HaohanWang/ImageNet-Sketch
* Paper: "Learning Robust Global Representations by Penalizing Local Predictive Power" - https://arxiv.org/abs/1905.13549

50,000 non photographic (or photos of such) images (sketches, doodles, mostly monochromatic) covering all 1000 ImageNet classes.

### ImageNet-Adversarial - [`results-imagenet-a.csv`](results-imagenet-a.csv)

* Source: https://github.com/hendrycks/natural-adv-examples
* Paper: "Natural Adversarial Examples" - https://arxiv.org/abs/1907.07174

A collection of 7500 images covering 200 of the 1000 ImageNet classes. Images are naturally occuring adversarial examples that confuse typical ImageNet classifiers. This is a challenging dataset, your typical ResNet-50 will score 0% top-1.

## TODO
* Explore adding a reduced version of ImageNet-C (Corruptions) and ImageNet-P (Perturbations) from https://github.com/hendrycks/robustness. The originals are huge and image size specific.
