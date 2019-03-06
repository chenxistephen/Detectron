#!/bin/bash
set -eu
python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/OpenImage/retinanet_4gpu_R-101-FPN_1x.yaml \
    OUTPUT_DIR ./detectron-output \
    USE_NCCL True
