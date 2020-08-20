#!/bin/bash
set -eu

GPUS=$1
NUM_GPUS=$2
MODEL_NAME=Final_Candidate
#Final_Candidate #IU_FP32 # Open800k_580/retr50_bs_4-iter_450000-lr_0.01 #$3
SCALE=400 #$7 # TEST SCALE
SOFTNMS=False
NMS_TH=0.5
MODEL_PATH=/media/data/chnxi/GOD/Models

TESTSETS="('bing5k_GOD', )"
#TESTSETS="('coco_2014_minival', )"
# TESTSETS="('GOD_FashionV2_val','GOD_furniture_val','GOD_coco_2014_minival','GOD_OpenImage_val', )"

WEIGHTS=$MODEL_PATH/$MODEL_NAME/model_final.pkl
#https://dl.fbaipublicfiles.com/detectron/36768677/12_2017_baselines/retinanet_R-50-FPN_2x.yaml.08_30_38.sgZIQZQ5/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl

#MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"

echo $MODEL_NAME
echo "Test Scale =" $SCALE
echo "Use SoftNMS =" $SOFTNMS
echo "NMS_TH = " $NMS_TH
echo "TESTSETS = " $TESTSETS
echo "GPUS = " $GPUS



CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
    --cfg configs/GOD/retinanet_R-50-FPN_8gpu.yaml \
    --eval_test \
    --compute_loc_pr \
    TEST.WEIGHTS "$WEIGHTS" \
    OUTPUT_DIR $MODEL_PATH/$MODEL_NAME/Test_$SCALE-SoftNMS-$SOFTNMS \
    TEST.SOFT_NMS.ENABLED $SOFTNMS \
    TEST.DATASETS "$TESTSETS" \
    NUM_GPUS $NUM_GPUS \
    TEST.SCALE $SCALE \
    TEST.NMS $NMS_TH \
    USE_NCCL True \
    | tee $MODEL_PATH/$MODEL_NAME/eval.log

    #--eval_test \
    #--multi-gpu-testing \
    # --compute_loc_pr \
    #TEST.DATASETS "('GOD_FashionV2_val','GOD_furniture_val','GOD_coco_2014_minival','GOD_OpenImage_val')" \
    #TEST.WEIGHTS $MODEL_PATH/$MODEL_NAME/model_final.pkl

