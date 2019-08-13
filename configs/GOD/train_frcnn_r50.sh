#!/bin/bash
set -eu
# GPUS=$1

BASE_LR=0.025 #0.02
IMS_PER_BATCH=4 #2
MAX_ITER=600000 # 2000000
TRAINSET=GOD_Open40k_train


# BASE_LR=0.02
# IMS_PER_BATCH=2
# MAX_ITER=2000000

JOB_NAME=frcnn_bs_$IMS_PER_BATCH-iter_$MAX_ITER-lr_$BASE_LR

echo $JOB_NAME

# CUDA_VISIBLE_DEVICES="$GPUS" 
python tools/train_net.py \
    --cfg configs/GOD/e2e_faster_rcnn_R-50-FPN_8gpu.yaml \
    SOLVER.BASE_LR $BASE_LR \
    TRAIN.DATASETS "('$TRAINSET', )" \
    SOLVER.MAX_ITER  $MAX_ITER \
    TRAIN.IMS_PER_BATCH $IMS_PER_BATCH \
    OUTPUT_DIR /data/chnxi/GOD/Models/Open40k/$JOB_NAME/ \
    | tee logs/god_$JOB_NAME.log
