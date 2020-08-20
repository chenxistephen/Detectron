#!/bin/bash
set -eu
GPUS=$1
NUM_GPUS=$2
SCALE=400 #$7 # TEST SCALE

SOFTNMS=False

MODEL_PATH=/media/data/chnxi/Detectron_Trained_Models/HomeFurniture/V1_Shipping
FINAL_MODEL=$MODEL_PATH/model_final.pkl


CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/HomeFurniture/hf_fashion_vi_retinanet.yaml \
    --multi-gpu-testing \
    --eval_test \
    TEST.WEIGHTS $FINAL_MODEL \
    OUTPUT_DIR $MODEL_PATH/Test_$SCALE-SoftNMS-$SOFTNMS \
    TEST.DATASETS "('furniture_val', )" \
    TEST.SOFT_NMS.ENABLED $SOFTNMS \
    NUM_GPUS $NUM_GPUS \
    TEST.SCALE $SCALE
    #--eval_test \
    #--multi-gpu-testing \
