#!/bin/bash
set -eu
GPUS=$1
ITER=$2
LR=$3
LOSS_ALPHA=0.25 #0.5 #0.25
LOSS_GAMMA=2.0
TRAINSET=Fashion_VI_train


CUDA_VISIBLE_DEVICES="$GPUS" python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/FashionV3/retinanet_R-50-FPN_2x_gn_4gpu.yaml \
    NUM_GPUS 4 \
    SOLVER.BASE_LR $LR \
    SOLVER.MAX_ITER $ITER \
    TRAIN.DATASETS "('$TRAINSET', )"
    RETINANET.LOSS_ALPHA $LOSS_ALPHA \
    RETINANET.LOSS_GAMMA $LOSS_GAMMA \
    OUTPUT_DIR ./detectron-output/$TRAINSET-GN_4gpu_iter_$ITER-lr_$LR-alpha_$LOSS_ALPHA-gamma_$LOSS_GAMMA \
    USE_NCCL True | tee logs/FashionV3/$TRAINSET-GN_4gpu_iter_$ITER-lr_$LR-alpha_$LOSS_ALPHA-gamma_$LOSS_GAMMA.log
