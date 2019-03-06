#!/bin/bash
set -eu
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/OpenImage/retinanet_1gpu_R-50-FPN_1x.yaml \
    OUTPUT_DIR ./detectron-output \
    USE_NCCL True
