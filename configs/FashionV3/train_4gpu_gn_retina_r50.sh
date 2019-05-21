#!/bin/bash
set -eu
GPUS=$1

CUDA_VISIBLE_DEVICES="$GPUS" python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/FashionV3/retinanet_R-50-FPN_2x_gn.yaml \
    NUM_GPUS 4 \
    SOLVER.BASE_LR 0.005 \
    OUTPUT_DIR ./detectron-output/gn \
    USE_NCCL True
