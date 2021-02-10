# Scripts
A train, validation, inference, and checkpoint cleaning script included in the github root folder. Scripts are not currently packaged in the pip release.

The training and validation scripts evolved from early versions of the [PyTorch Imagenet Examples](https://github.com/pytorch/examples). I have added significant functionality over time, including CUDA specific performance enhancements based on
[NVIDIA's APEX Examples](https://github.com/NVIDIA/apex/tree/master/examples).

## Training Script

The variety of training args is large and not all combinations of options (or even options) have been fully tested. For the training dataset folder, specify the folder to the base that contains a `train` and `validation` folder.

To train an SE-ResNet34 on ImageNet, locally distributed, 4 GPUs, one process per GPU w/ cosine schedule, random-erasing prob of 50% and per-pixel random value:

`./distributed_train.sh 4 /data/imagenet --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 --amp -j 4`

NOTE: It is recommended to use PyTorch 1.7+ w/ PyTorch native AMP and DDP instead of APEX AMP. `--amp` defaults to native AMP as of timm ver 0.4.3.  `--apex-amp` will force use of APEX components if they are installed.
 
## Validation / Inference Scripts

Validation and inference scripts are similar in usage. One outputs metrics on a validation set and the other outputs topk class ids in a csv. Specify the folder containing validation images, not the base as in training script. 

To validate with the model's pretrained weights (if they exist):

`python validate.py /imagenet/validation/ --model seresnext26_32x4d --pretrained`

To run inference from a checkpoint:

`python inference.py /imagenet/validation/ --model mobilenetv3_large_100 --checkpoint ./output/train/model_best.pth.tar`