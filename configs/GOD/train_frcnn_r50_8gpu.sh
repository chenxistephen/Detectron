#!/bin/bash
set -eu
# GPUS=$1

BASE_LR=0.03 #0.02
IMS_PER_BATCH=4 #2
MAX_ITER=800000 # 2000000
TRAINSET=GOD_Open800k_train


# BASE_LR=0.02
# IMS_PER_BATCH=2
# MAX_ITER=2000000

JOB_NAME=frcnn_8gpuxbs$IMS_PER_BATCH-iter_$MAX_ITER-lr_$BASE_LR
OUTPUT_DIR=/ssddata-multimedia/chnxi/GOD/Models/$TRAINSET/$JOB_NAME/
mkdir -p $OUTPUT_DIR
echo $JOB_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python tools/train_net.py \
    --cfg configs/GOD/e2e_faster_rcnn_R-50-FPN_8gpu.yaml \
    --multi-gpu-testing \
    SOLVER.BASE_LR $BASE_LR \
    TRAIN.DATASETS "('$TRAINSET', )" \
    SOLVER.MAX_ITER  $MAX_ITER \
    TRAIN.IMS_PER_BATCH $IMS_PER_BATCH \
    OUTPUT_DIR $OUTPUT_DIR \
    | tee logs/$TRAINSET/$JOB_NAME.log
