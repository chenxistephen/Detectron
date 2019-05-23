#!/bin/bash
set -eu
GPUS=$1

CUDA_VISIBLE_DEVICES="$GPUS" python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/FashionV3/fashion_vi_110_retinanet.yaml \
    OUTPUT_DIR ./detectron-output \
    USE_NCCL True
