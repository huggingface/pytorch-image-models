#!/usr/bin/env bash
source /workspace/venv/bin/activate

pip install --upgrade pip
pip install -r requirements-sotabench.txt

apt-get update
apt-get install -y libjpeg-dev zlib1g-dev libpng-dev libwebp-dev
pip uninstall -y pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# FIXME this shouldn't be needed but sb dataset upload functionality doesn't seem to work
apt-get install wget
#wget -q https://onedrive.hyper.ai/down/ImageNet/data/ImageNet2012/ILSVRC2012_devkit_t12.tar.gz -P ./.data/vision/imagenet
wget -q https://onedrive.hyper.ai/down/ImageNet/data/ImageNet2012/ILSVRC2012_img_val.tar -P ./.data/vision/imagenet
