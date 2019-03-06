#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
TESTATT=OpenImage_train # $2 #furniture_val_SearchIntent
GPUS=$1
SCALE=600 #$7 # TEST SCALE


#MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"



CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/OpenImage/retinanet_1gpu_R-50-FPN_1x.yaml \
    TEST.WEIGHTS Trained_Models/HomeFurniture/V1_Shipping/model_final.pkl \
    OUTPUT_DIR backfill/HF_Detector/OpenImage/ \
    TEST.DATASETS "('$TESTATT',)" \
    --multi-gpu-testing \
    NUM_GPUS 4 \
    TEST.SCALE $SCALE \
    | tee logs/run_HF_on_$TESTATT.log
