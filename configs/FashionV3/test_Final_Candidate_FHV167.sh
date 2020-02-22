#!/bin/bash
set -eu
GPUS=$1
NUM_GPUS=$2
SCALE=400 #$7 # TEST SCALE

SOFTNMS=False

FINAL_MODEL=/media/data/chnxi/FashionV3/Models/Final_Shipping_Candidate/model_final.pkl


CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/FashionV3/fashion_hf_vi_167_retinanet.yaml \
    --eval_test \
    --multi-gpu-testing \
    TEST.WEIGHTS $FINAL_MODEL \
    OUTPUT_DIR $MODEL_PATH/$MODEL_NAME/Test_$SCALE-SoftNMS-$SOFTNMS \
    TEST.DATASETS "('FashionV2_val', )" \
    TEST.SOFT_NMS.ENABLED $SOFTNMS \
    NUM_GPUS $NUM_GPUS \
    TEST.SCALE $SCALE