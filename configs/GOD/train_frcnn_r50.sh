#!/bin/bash
set -eu
# GPUS=$1
JOB_NAME=batchsize4

# CUDA_VISIBLE_DEVICES="$GPUS" 
python tools/train_net.py \
    --cfg configs/GOD/e2e_faster_rcnn_R-50-FPN_8gpu.yaml \
    OUTPUT_DIR ./detectron-output \
    USE_NCCL True \
    | tee logs/train_$JOB_NAME.log
