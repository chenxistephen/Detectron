#!/bin/bash
set -eu
#MODEL_NAME=$1 #iter_90000_alpha_0.25_gamma_2.0_lr_0.01
#TESTATT=('GOD_FashionV2_val','GOD_furniture_val','GOD_coco_2014_minival','GOD_OpenImage_val') # $2 #furniture_val_SearchIntent
GPUS=$1
# NUM_GPUS=$2
MODEL_NAME=frcnn101_4-gpuxbs4-iter_400000-lr_0.02 #$2 #god_frcnn_tuneiter_1000000_try6_nonccl
SCALE=400 #$7 # TEST SCALE

MODEL_PATH=/media/data/chnxi/VisualIntentV3_MightyAI/Models/


#MODEL_NAME=scale"$TRAINSCALE"_iter90000_alpha"$ALPHA"_gamma"$GAMMA"_lr"$LR"_body"$BODY"

echo $MODEL_NAME


# CUDA_VISIBLE_DEVICES="$GPUS" python tools/test_net.py \
#     --cfg configs/GOD/e2e_faster_rcnn_R-50-FPN_8gpu.yaml \
#     --eval_test \
#     --multi-gpu-testing \
#     TEST.WEIGHTS $MODEL_PATH/$MODEL_NAME/model_final.pkl \
#     OUTPUT_DIR $MODEL_PATH/$MODEL_NAME/Test_$SCALE \
#     TEST.DATASETS "('GOD_FashionV2_val','GOD_furniture_val','GOD_coco_2014_minival','GOD_OpenImage_val')" \
#     NUM_GPUS $NUM_GPUS \
#     TEST.SCALE $SCALE \
#     USE_NCCL True \
#     | tee logs/eval_$MODEL_NAME-Test_$SCALE.log
#     #--range 1000 2000 \


CUDA_VISIBLE_DEVICES="$GPUS" python tools/infer_detection_bbox.py \
    --cfg configs/VisualIntent_GOD/e2e_faster_rcnn_R-101-FPN_8gpu.yaml \
    --output-dir $MODEL_PATH/$MODEL_NAME/visualizations/CameraV3 \
    --image-ext jpg \
    --im_or_folder /media/data/chnxi/CameraMeasurementSetV3 \
    --im_list /media/data/chnxi/CameraMeasurementSetV3/camera_set_v3_imglist.tsv \
    --wts $MODEL_PATH/$MODEL_NAME/model_final.pkl \
    --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_taxonomy_20191017.tsv