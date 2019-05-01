#!/bin/bash
set -eu
GPUS=$1

CUDA_VISIBLE_DEVICES="$GPUS" python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/GOD/e2e_faster_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR ./detectron-output \
    USE_NCCL True
