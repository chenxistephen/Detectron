#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
TESTATT=logo_1048_val # $2 #furniture_val_SearchIntent
GPUS=$1
TRAINSCALE=$2
ROIRES=$3
SAMPLERATIO=$4
RPNRATIONUM=$5
SCALE=$6 # TEST SCALE

RPNRATIO="0.5, 1.0, 2.0"

MODEL_NAME=frcnn_scale"$TRAINSCALE"_roires"$ROIRES"_sampleratio"$SAMPLERATIO"_rpnratio"$RPNRATIONUM"

echo $MODEL_NAME

if [ "$RPNRATIONUM" = "3" ]
then
    RPNRATIO="0.333, 0.5, 1.0, 2.0, 3.0"
fi

echo $RPNRATIO

CUDA_VISIBLE_DEVICES="$GPUS" python2 tools/test_net.py \
    --cfg configs/Logo/e2e_faster_rcnn_R-50-FPN_1x.yaml \
    TEST.WEIGHTS Trained_Models/Logo/$MODEL_NAME/model_final.pkl \
    OUTPUT_DIR detectron-output/Logo/$MODEL_NAME/Test_$SCALE \
    TEST.DATASETS "('$TESTATT',)" \
    NUM_GPUS 1 \
    TEST.SCALE $SCALE \
    FAST_RCNN.ROI_XFORM_RESOLUTION $ROIRES \
    FAST_RCNN.ROI_XFORM_SAMPLING_RATIO $SAMPLERATIO \
    FPN.RPN_ASPECT_RATIOS "($RPNRATIO)" \
    | tee eval_$MODEL_NAME-Test_$SCALE.log
