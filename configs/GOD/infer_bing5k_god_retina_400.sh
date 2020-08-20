#!/bin/bash
set -eu

GPUS=0
SCALE=400 #$7 # TEST SCALE

DETECTRON_PATH=/home/chnxi/Detectron


CUDA_VISIBLE_DEVICES="$GPUS" python $DETECTRON_PATH/tools/infer_detection_bbox.py \
    --cfg retinanet_R-50-FPN_8gpu.yaml \
    --output-dir inference/visuslizations/bing5k \
    --image-ext jpg \
    --im_or_folder /media/data/chnxi/BingMeasurement_5k/Images \
    --wts model_final.pkl \
    --class_list_file /media/data/chnxi/GOD/taxonomy/GODv1_May2020_579_Labels_final.txt