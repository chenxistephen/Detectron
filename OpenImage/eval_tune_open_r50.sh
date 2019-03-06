#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
TESTATT=OpenImage_val # $2 #furniture_val_SearchIntent
GPUS=$1
TRAINSCALE=$2
ALPHA=$3
GAMMA=$4
LR=$5
BODY=$6 
SCALE=$7 # TEST SCALE


MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"

echo $MODEL_NAME


CUDA_VISIBLE_DEVICES="$GPUS" python2 tools/test_net.py \
    --cfg configs/OpenImage/retinanet_1gpu_R-50-FPN_1x.yaml \
    TEST.WEIGHTS Trained_Models/OpenImage/$MODEL_NAME/model_final.pkl \
    OUTPUT_DIR detectron-output/OpenImage/$MODEL_NAME/Test_$SCALE \
    TEST.DATASETS "('$TESTATT',)" \
    NUM_GPUS 1 \
    TEST.SCALE $SCALE \
    | tee eval_$MODEL_NAME-Test_$SCALE.log
