#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
TESTATT=OpenImage_train # $2 #furniture_val_SearchIntent
GPUS=$1
SCALE=600 #$7 # TEST SCALE
NUM_GPU=$2
OUTPUT_DIR=backfill/Object365_Detector/$TESTATT/
#MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi

CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/Object365/e2e_faster_rcnn_R-101-FPN_2x.yaml \
    --multi-gpu-testing \
    TEST.WEIGHTS /media/data/chnxi/Object365/Models/frcnn101_4-gpuxbs4-iter_1350000-lr_0.02/model_final.pkl \
    OUTPUT_DIR $OUTPUT_DIR \
    TEST.DATASETS "('$TESTATT',)" \
    NUM_GPUS $NUM_GPU \
    TEST.SCALE $SCALE \
    TEST.MAX_SIZE 1000 \
    | tee $OUTPUT_DIR/run_HF_on_$TESTATT.log
