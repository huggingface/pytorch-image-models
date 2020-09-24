#!/usr/bin/env bash
source /workspace/venv/bin/activate

pip install -r requirements-sotabench.txt

pip uninstall -y pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# FIXME this shouldn't be needed but sb dataset upload functionality doesn't seem to work
apt-get install wget
wget https://onedrive.hyper.ai/down/ImageNet/data/ImageNet2012/ILSVRC2012_devkit_t12.tar.gz -P ./.data/vision/imagenet
wget https://onedrive.hyper.ai/down/ImageNet/data/ImageNet2012/ILSVRC2012_img_val.tar -P ./.data/vision/imagenet
