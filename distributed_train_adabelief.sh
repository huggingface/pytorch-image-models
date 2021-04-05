#!/bin/bash
NUM_PROC=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@" --model efficientnet_b0 --weight-decay 2.5e-2 --drop 0.2 --drop-path 0.2 --lr 0.002 --batch-size 192 --epochs 400 --sched cosine --opt adabelief --workers 8 --warmup-lr 1e-4 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --bn-momentum 0.1 --mixup 0.2 --mixup-off-epoch 400 --min-lr 1e-5 
