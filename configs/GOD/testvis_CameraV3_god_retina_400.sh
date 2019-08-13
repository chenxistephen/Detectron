#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
#TESTATT=('GOD_FashionV2_val','GOD_furniture_val','GOD_coco_2014_minival','GOD_OpenImage_val') # $2 #furniture_val_SearchIntent
GPUS=$1
NUM_GPUS=$2
MODEL_NAME=$3 #god_frcnn_tuneiter_1000000_try6_nonccl
SCALE=400 #$7 # TEST SCALE

MODEL_PATH=/media/data/chnxi/GOD/Models
OUTPUT_DIR=$MODEL_PATH/$MODEL_NAME/Test_$SCALE


#MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"

echo $MODEL_NAME


# CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
#     --cfg configs/GOD/retinanet_R-50-FPN_8gpu.yaml \
#     --multi-gpu-testing \
#     TEST.WEIGHTS $MODEL_PATH/$MODEL_NAME/model_final.pkl \
#     OUTPUT_DIR $OUTPUT_DIR \
#     TEST.DATASETS "('CameraV3_GOD', 'bing5k_GOD')" \
#     NUM_GPUS $NUM_GPUS \
#     TEST.SCALE $SCALE \
#     USE_NCCL True \
#     | tee logs/eval_$MODEL_NAME-Test_$SCALE.log
    #--range 1000 2000 \
    #--multi-gpu-testing \
    #--eval_test \

#########################################################
TESTSET=CameraV3_GOD

python tools/visualize_results.py --dataset $TESTSET --thresh 0.5 --detections $OUTPUT_DIR/$TESTSET/retinanet/detections.pkl --output-dir $OUTPUT_DIR/visualizations/$TESTSET/ --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_V1_labels.txt


#########################################################
TESTSET=bing5k_GOD

python tools/visualize_results.py --dataset $TESTSET --thresh 0.5 --detections $OUTPUT_DIR/$TESTSET/retinanet/detections.pkl --output-dir $OUTPUT_DIR/visualizations/$TESTSET/ --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_V1_labels.txt