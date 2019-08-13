#!/bin/bash
set -eu
GPUS=$1
ITER=$2
LR=$3
LOSS_ALPHA=$4 #0.5 #0.25
LOSS_GAMMA=2.0


CUDA_VISIBLE_DEVICES="$GPUS" python tools/train_net.py \
    --cfg configs/FashionV3/fashion_hf_vi_167_retr50_2x_4gpu.yaml \
    NUM_GPUS 4 \
    SOLVER.BASE_LR $LR \
    SOLVER.MAX_ITER $ITER \
    RETINANET.LOSS_ALPHA $LOSS_ALPHA \
    RETINANET.LOSS_GAMMA $LOSS_GAMMA \
    OUTPUT_DIR ./detectron-output/fhv_4gpu_iter_$ITER-lr_$LR-alpha_$LOSS_ALPHA-gamma_$LOSS_GAMMA \
    USE_NCCL True | tee logs/FashionV3/fhv_167_4gpu_iter_$ITER-lr_$LR-alpha_$LOSS_ALPHA-gamma_$LOSS_GAMMA.log
