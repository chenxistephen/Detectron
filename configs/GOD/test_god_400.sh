#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
#TESTATT=('GOD_FashionV2_val','GOD_furniture_val','GOD_OpenImage_val','GOD_coco_2014_minival') # $2 #furniture_val_SearchIntent
GPUS=$1
NUM_GPUS=$2
MODEL_NAME=$3 #god_frcnn_tuneiter_200000_2.0
SCALE=400 #$7 # TEST SCALE

MODEL_PATH=/media/data/chnxi/GOD/Models


#MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"

echo $MODEL_NAME


CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/GOD/e2e_faster_rcnn_R-50-FPN_8gpu.yaml \
    --eval_test \
    TEST.WEIGHTS $MODEL_PATH/$MODEL_NAME/model_final.pkl \
    OUTPUT_DIR $MODEL_PATH/$MODEL_NAME/test/Test_$SCALE \
    TEST.DATASETS "('GOD_OpenImage_val', )" \
    NUM_GPUS $NUM_GPUS \
    TEST.SCALE $SCALE \
    | tee logs/eval_$MODEL_NAME-Test_$SCALE.log
