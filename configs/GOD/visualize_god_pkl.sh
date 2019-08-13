#!/bin/bash
set -eu

MODEL_NAME=Open800K/retr50_bs_2-iter_900000-lr_0.01
#retr50_bs_2-iter_1600000-lr_0.01
MODEL_PATH=/media/data/chnxi/GOD/Models/$MODEL_NAME/Test_400-SoftNMS-True
#god_frcnn_tuneiter_1000000_try6_nonccl

# python tools/visualize_results.py --dataset GOD_OpenImage_val --thresh 0.5 --detections /media/data/chnxi/GOD/Models/$MODEL_NAME/Test_400/GOD_OpenImage_val/generalized_rcnn/detection_range_1000_2000.pkl --output-dir visualizations/GOD/$MODEL_NAME --range 1000 2000 --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_V1_labels.txt

################################
TESTSET=GOD_coco_2014_minival

python tools/visualize_results.py --dataset $TESTSET --thresh 0.5 --detections $MODEL_PATH/$TESTSET/retinanet/detections.pkl --output-dir $MODEL_PATH/visualizations/$TESTSET/ --sampleNum 200 --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_V1_labels.txt


################################
TESTSET=GOD_OpenImage_val

# python tools/visualize_results.py --dataset $TESTSET --thresh 0.5 --detections $MODEL_PATH/$TESTSET/retinanet/detections.pkl --output-dir $MODEL_PATH/visualizations/$TESTSET/ --sampleNum 200 --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_V1_labels.txt

################################
TESTSET=GOD_FashionV2_val

# python tools/visualize_results.py --dataset $TESTSET --thresh 0.5 --detections $MODEL_PATH/$TESTSET/retinanet/detections.pkl --output-dir $MODEL_PATH/visualizations/$TESTSET/ --sampleNum 200 --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_V1_labels.txt

################################
TESTSET=GOD_furniture_val

# python tools/visualize_results.py --dataset $TESTSET --thresh 0.5 --detections $MODEL_PATH/$TESTSET/retinanet/detections.pkl --output-dir $MODEL_PATH/visualizations/$TESTSET/ --sampleNum 200 --class_list_file /media/data/chnxi/GOD/taxonomy/GOD_V1_labels.txt