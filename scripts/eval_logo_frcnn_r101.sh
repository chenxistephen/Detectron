#!/bin/bash
set -eu
MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
TESTATT=logo_1048_val # $2 #furniture_val_SearchIntent
GPUS=$2
SCALE=$3

echo "SCALE" $SCALE

CUDA_VISIBLE_DEVICES="$GPUS" python2 tools/test_net.py \
    --cfg configs/Logo/e2e_faster_rcnn_R-101-FPN_1x.yaml \
    TEST.WEIGHTS Trained_Models/Logo/$MODEL_NAME/model_final.pkl \
    OUTPUT_DIR detectron-output/Logo/$MODEL_NAME/Test_$SCALE \
    TEST.DATASETS "('$TESTATT',)" \
    NUM_GPUS 1 \
    TEST.SCALE $SCALE \
    | tee eval_$MODEL_NAME-Test_$SCALE.log
