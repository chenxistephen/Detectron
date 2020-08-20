#!/bin/bash
set -eu

GPUS=$1
NUM_GPUS=1
SCALE=400
SOFTNMS=False

MODEL_NAME=Final_Shipping_Candidate
#Fashion_HF_VI/bs2-iter_300000-lr_0.01-alpha_0.25-gamma_2
#fashionv3_retr50_iter360000_lr0.005_alpha0.5_gamma2.0
###############################################################################################
MODEL_PATH=/media/data/chnxi/FashionV3/Models/


CTH=$MODEL_PATH/$MODEL_NAME/Test_$SCALE-SoftNMS-$SOFTNMS/FashionV2_val/retinanet/classwise_pr_curves.pkl

echo "CTH = " $CTH
###############################################################################################

CUDA_VISIBLE_DEVICES="$GPUS" python tools/infer_detection_bbox.py \
    --cfg configs/FashionV3/fashion_vi_110_retinanet.yaml \
    --wts $MODEL_PATH/$MODEL_NAME/model_final.pkl \
    --thresh 0.45 \
    --image-ext jpg \
    --im_or_folder /media/data/chnxi/FashionCeleb_Salient/Images \
    --output-dir $MODEL_PATH/$MODEL_NAME/visualizations/FashionCeleb_NoSoftNMS \
    --class_list_file /media/data/chnxi/FashionV3/taxonomy/fashion_furniture_visualintent_166_labels.txt \
    TEST.SOFT_NMS.ENABLED False \
    TEST.SCALE $SCALE \

# CUDA_VISIBLE_DEVICES="$GPUS" python tools/infer_detection_bbox.py \
#     --cfg configs/GOD/e2e_faster_rcnn_R-50-FPN_8gpu.yaml \
#     --output-dir $MODEL_PATH/$MODEL_NAME/visualizations/CameraV3 \
#     --image-ext jpg \
#     --im_or_folder /media/data/chnxi/CameraMeasurementSetV3 \
#     --im_list /media/data/chnxi/CameraMeasurementSetV3/camera_set_v3_imglist.tsv \
#     --wts $MODEL_PATH/$MODEL_NAME/model_final.pkl \
#     --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_V1_labels.txt