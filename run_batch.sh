#!/usr/bin/env bash

# Input: a text file, each line is a model name registered with `timm`
# Sequentially train these models with `imagenet_100cls`

input_file=$1
[[ ! -f $input_file ]] && echo "${input_file} does not exists" && exit 1

imgnet100_path=${2:-/nas/common_data/imagenet_100cls}
[[ ! -d $imgnet100_path ]] && echo "Path to imagenet_100cls: ${imgnet100_path} not found" && exit 1

LOG_DIR="./logs"
COMMON_ARGS="""
    --data-dir ${imgnet100_path} \
    --num-classes 100 \
    -b 64 \
    --epochs 3 \
    -j 8 \
    --seed 42 \
    --checkpoint-hist 2 \
    --log-interval 100 \
"""

mkdir -p $LOG_DIR

# Install MLFlow
pip install mlflow
# Start MLFlow server in the background
mlflow server &
# Wait for the serer to start (5 secs)
sleep 5

while read model; do
    # echo $model
    /usr/bin/env python3 train.py \
        --model $model \
        $COMMON_ARGS \
        2>&1 | tee ${LOG_DIR}/${model}.log
done < $input_file

# Stop the MLFlow server at the end
pkill -f mlflow