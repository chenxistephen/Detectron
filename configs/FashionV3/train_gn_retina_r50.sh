#!/bin/bash
set -eu
GPUS=$1

CUDA_VISIBLE_DEVICES="$GPUS" python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/FashionV3/retinanet_R-50-FPN_2x_gn.yaml \
    OUTPUT_DIR ./detectron-output \
    USE_NCCL True
