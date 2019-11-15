#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
#TESTATT=('GOD_FashionV2_val','GOD_furniture_val','GOD_coco_2014_minival','GOD_OpenImage_val') # $2 #furniture_val_SearchIntent
GPUS=$1
NUM_GPUS=$2
MODEL_NAME=$3 #god_frcnn_tuneiter_1000000_try6_nonccl
SCALE=400 #$7 # TEST SCALE
SOFTNMS=False
VIS=False
MODEL_PATH=/media/data/chnxi/Object365/Models


#MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"

echo $MODEL_NAME
echo "Test Scale =" $SCALE
echo "Use SoftNMS =" $SOFTNMS

CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/Object365/e2e_faster_rcnn_R-101-FPN_2x.yaml \
    --eval_test \
    --multi-gpu-testing \
    MODEL.NUM_CLASSES 366 \
    VIS $VIS \
    VIS_TH 0.5 \
    TEST.WEIGHTS $MODEL_PATH/$MODEL_NAME/model_iter99999.pkl \
    OUTPUT_DIR $MODEL_PATH/$MODEL_NAME/Test_$SCALE-SoftNMS-$SOFTNMS \
    TEST.DATASETS "('Object365_val', )" \
    TEST.SOFT_NMS.ENABLED $SOFTNMS \
    NUM_GPUS $NUM_GPUS \
    TEST.SCALE $SCALE \
    USE_NCCL True \
    | tee logs/GOD/eval_$MODEL_NAME-Test_$SCALE-SoftNMS-$SOFTNMS.log
    #--range 1000 2000 \
    #--compute_loc_pr\
