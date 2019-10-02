#!/bin/bash
set -eu
GPUS=$1

BASE_LR=$2 #0.025 #0.02
MAX_ITER=1600000 # 2000000
IMS_PER_BATCH=4 #2
TRAINSET=GOD_Open800k_train


# BASE_LR=0.02
# IMS_PER_BATCH=2
# MAX_ITER=2000000

JOB_NAME=frcnn_4gpuxbs$IMS_PER_BATCH-iter_$MAX_ITER-lr_$BASE_LR
OUTPUT_DIR=/data/users/chnxi/GOD/Models/$TRAINSET/$JOB_NAME/
mkdir $OUTPUT_DIR

echo $JOB_NAME

# CUDA_VISIBLE_DEVICES="$GPUS" 
python tools/train_net.py \
    --cfg configs/GOD/e2e_faster_rcnn_R-50-FPN_8gpu.yaml \
    SOLVER.BASE_LR $BASE_LR \
    NUM_GPUS 4 \
    TRAIN.DATASETS "('$TRAINSET', )" \
    SOLVER.MAX_ITER  $MAX_ITER \
    TRAIN.IMS_PER_BATCH $IMS_PER_BATCH \
    OUTPUT_DIR $OUTPUT_DIR\
    | tee logs/$TRAINSET/$JOB_NAME.log