# Training Examples

## EfficientNet-B2 with RandAugment - 80.4 top-1, 95.1 top-5
These params are for dual Titan RTX cards with NVIDIA Apex installed:

`./distributed_train.sh 2 /imagenet/ --model efficientnet_b2 -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016`

## MixNet-XL with RandAugment - 80.5 top-1, 94.9 top-5
This params are for dual Titan RTX cards with NVIDIA Apex installed:

`./distributed_train.sh 2 /imagenet/ --model mixnet_xl -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .969 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.3 --amp --lr .016 --dist-bn reduce`

## SE-ResNeXt-26-D and SE-ResNeXt-26-T
These hparams (or similar) work well for a wide range of ResNet architecture, generally a good idea to increase the epoch # as the model size increases... ie approx 180-200 for ResNe(X)t50, and 220+ for larger. Increase batch size and LR proportionally for better GPUs or with AMP enabled. These params were for 2 1080Ti cards:

`./distributed_train.sh 2 /imagenet/ --model seresnext26t_32x4d --lr 0.1 --warmup-epochs 5 --epochs 160 --weight-decay 1e-4 --sched cosine --reprob 0.4 --remode pixel -b 112`

## EfficientNet-B3 with RandAugment - 81.5 top-1, 95.7 top-5
The training of this model started with the same command line as EfficientNet-B2 w/ RA above. After almost three weeks of training the process crashed. The results weren't looking amazing so I resumed the training several times with tweaks to a few params (increase RE prob, decrease rand-aug, increase ema-decay). Nothing looked great. I ended up averaging the best checkpoints from all restarts. The result is mediocre at default res/crop but oddly performs much better with a full image test crop of 1.0. 

## EfficientNet-B0 with RandAugment - 77.7 top-1, 95.3 top-5
[Michael Klachko](https://github.com/michaelklachko) achieved these results with the command line for B2 adapted for larger batch size, with the recommended B0 dropout rate of 0.2.

`./distributed_train.sh 2 /imagenet/ --model efficientnet_b0 -b 384 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048`

## ResNet50 with JSD loss and RandAugment (clean + 2x RA augs) - 79.04 top-1, 94.39 top-5

Trained on two older 1080Ti cards, this took a while. Only slightly, non statistically better ImageNet validation result than my first good AugMix training of 78.99. However, these weights are more robust on tests with ImageNetV2, ImageNet-Sketch, etc. Unlike my first AugMix runs, I've enabled SplitBatchNorm, disabled random erasing on the clean split, and cranked up random erasing prob on the 2 augmented paths.

`./distributed_train.sh 2 /imagenet -b 64 --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce`

## EfficientNet-ES (EdgeTPU-Small) with RandAugment - 78.066 top-1, 93.926 top-5
Trained by [Andrew Lavin](https://github.com/andravin) with 8 V100 cards. Model EMA was not used, final checkpoint is the average of 8 best checkpoints during training.

`./distributed_train.sh 8 /imagenet --model efficientnet_es -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2  --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064`

## MobileNetV3-Large-100 - 75.766 top-1, 92,542 top-5

`./distributed_train.sh 2 /imagenet/ --model mobilenetv3_large_100 -b 512 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064 --lr-noise 0.42 0.9`


## ResNeXt-50 32x4d w/ RandAugment - 79.762 top-1, 94.60 top-5
These params will also work well for SE-ResNeXt-50 and SK-ResNeXt-50 and likely 101. I used them for the SK-ResNeXt-50 32x4d that I trained with 2 GPU using a slightly higher LR per effective batch size (lr=0.18, b=192 per GPU). The cmd line below are tuned for 8 GPU training.


`./distributed_train.sh 8 /imagenet --model resnext50_32x4d --lr 0.6 --warmup-epochs 5 --epochs 240 --weight-decay 1e-4 --sched cosine --reprob 0.4 --recount 3 --remode pixel --aa rand-m7-mstd0.5-inc1 -b 192 -j 6 --amp --dist-bn reduce`

