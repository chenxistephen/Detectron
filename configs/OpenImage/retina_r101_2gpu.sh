#!/bin/bash
set -eu
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/OpenImage/retinanet_2gpu_R-101-FPN_1x.yaml \
    OUTPUT_DIR ./detectron-output \
    USE_NCCL True
