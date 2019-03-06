#!/bin/bash
set -eu
MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
TESTATT=logo_1048_val # $2 #furniture_val_SearchIntent
GPUS=$2
SCALE=$3

CUDA_VISIBLE_DEVICES="$GPUS" python2 tools/test_net.py \
    --cfg configs/Logo/logo_retinanet_1gpu_R-101-FPN_1x.yaml \
    TEST.WEIGHTS Trained_Models/Logo/$MODEL_NAME/model_final.pkl \
    OUTPUT_DIR detectron-output/Logo/$MODEL_NAME \
    TEST.DATASETS "('$TESTATT',)" \
    NUM_GPUS 1 \
    TEST.SCALE $SCALE \
    VIS False \
    VIS_TH 0.5
