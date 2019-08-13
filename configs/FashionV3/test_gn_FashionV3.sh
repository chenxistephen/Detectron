#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
#TEST.DATASETS=('FashionV2_val','bing5k_fashion', 'bing5k_fashion_thumbnail') # $2 #furniture_val_SearchIntent
GPUS=$1
NUM_GPUS=$2
MODEL_NAME=$3 #fashionv3_retr50_iter360000_lr0.005_alpha0.5_gamma2.0
SCALE=400 #$7 # TEST SCALE

SOFTNMS=True

MODEL_PATH=/media/data/chnxi/FashionV3/Models


#MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"

echo $MODEL_NAME
echo "Test Scale =" $SCALE


CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/FashionV3/retinanet_R-50-FPN_2x_gn.yaml \
    --eval_test \
    --multi-gpu-testing \
    TEST.WEIGHTS $MODEL_PATH/$MODEL_NAME/model_final.pkl \
    OUTPUT_DIR $MODEL_PATH/$MODEL_NAME/Test_$SCALE-SoftNMS-$SOFTNMS \
    TEST.DATASETS "('bing5k_fashion', )" \
    TEST.SOFT_NMS.ENABLED $SOFTNMS \
    NUM_GPUS $NUM_GPUS \
    TEST.SCALE $SCALE \
    | tee logs/eval_$MODEL_NAME-Test_$SCALE.log
    #--range $START_ID $END_ID \
    # TEST.DATASETS "('FashionV2_val', 'bing5k_fashion',)" \
