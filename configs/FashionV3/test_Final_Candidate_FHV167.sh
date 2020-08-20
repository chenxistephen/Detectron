#!/bin/bash
set -eu
GPUS=$1
NUM_GPUS=$2
SCALE=400 #$7 # TEST SCALE

SOFTNMS=False

MODEL_PATH=/media/data/chnxi/FashionV3/Models/Final_Shipping_Candidate
FINAL_MODEL=$MODEL_PATH/model_final.pkl


CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/FashionV3/fashion_hf_vi_167_retinanet.yaml \
    TEST.WEIGHTS $FINAL_MODEL \
    OUTPUT_DIR $MODEL_PATH/Test2_$SCALE-SoftNMS-$SOFTNMS \
    TEST.DATASETS "('FashionV2_val', )" \
    TEST.SOFT_NMS.ENABLED $SOFTNMS \
    NUM_GPUS $NUM_GPUS \
    TEST.SCALE $SCALE
    #--eval_test \
    #--multi-gpu-testing \
