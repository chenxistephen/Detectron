#!/bin/bash
set -eu
#GPUS=$1
ITER=300000 #$1
LR=0.01 #$2
LOSS_ALPHA=0.25 # $3
LOSS_GAMMA=2.0

IMS_PER_BATCH=2


JOB_NAME=fhv_8gpu_bs$IMS_PER_BATCH-iter_$ITER-lr_$LR-alpha_$LOSS_ALPHA-gamma_$LOSS_GAMMA

OUTPUT_DIR=/data/users/chnxi/FashionV3/Models/Fashion_HF_VI/$JOB_NAME/

python tools/train_net.py \
    --cfg configs/FashionV3/fashion_hf_vi_167_retr50_8gpu.yaml \
    NUM_GPUS 8 \
    SOLVER.BASE_LR $LR \
    SOLVER.MAX_ITER $ITER \
    RETINANET.LOSS_ALPHA $LOSS_ALPHA \
    RETINANET.LOSS_GAMMA $LOSS_GAMMA \
    TRAIN.IMS_PER_BATCH $IMS_PER_BATCH \
    OUTPUT_DIR $OUTPUT_DIR \
    USE_NCCL True | tee logs/FashionV3/$JOB_NAME.log
