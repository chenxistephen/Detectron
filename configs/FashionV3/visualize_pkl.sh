#!/bin/bash
set -eu

# MODEL_NAME=fashionv3_retr50_iter360000_lr0.005_alpha0.5_gamma2.0

# python tools/visualize_results.py --dataset FashionV2_val --thresh 0.547 --detections /media/data/chnxi/FashionV3/Models/$MODEL_NAME/Test_600/FashionV2_val/retinanet/detection_range_1000_2000.pkl --output-dir visualizations/FashionV3/$MODEL_NAME --range 1000 2000 --class_list_file /media/data/chnxi/FashionV2/taxonomy/fashion_intentbg_110_labels.txt



#DETPKL=/media/data/chnxi/FashionV3/Models/hf_fashion_vi_nd/Train_400-Test_400-SoftNMS-True/HFV_FashionV2_val/retinanet/detections.pkl

# python tools/visualize_results.py --dataset HFV_FashionV2_val --thresh 0.5 --detections $DETPKL --output-dir visualizations/FashionV3/$MODEL_NAME  --class_list_file /media/data/chnxi/HomeFurniture/taxonomy/furniture_fashion_visualintent_166_labels.txt

#DETPKL=/home/chnxi/FashionV3/Models/hf_fashion_vi_nd/400/detections.pkl

#DETPKL=/home/chnxi/FashionV3/Models/hf_fashion_vi_nd/Train_400-Test_400-SoftNMS-True/furniture_val/retinanet/detections.pkl

#python tools/visualize_results.py --dataset furniture_val --thresh 0.5 --detections $DETPKL --output-dir visualizations/HFV2/$MODEL_NAME  --class_list_file /media/data/chnxi/HomeFurniture/taxonomy/furniture_fashion_visualintent_166_labels.txt


# MODEL_NAME=hf_fashion_vi_nd
# DETPKL=/home/chnxi/FashionV3/Models/hf_fashion_vi_nd/Train_400-Test_400-SoftNMS-True/HFV_FashionV2_val/retinanet/detections.pkl
# python tools/visualize_results.py --dataset HFV_FashionV2_val --thresh 0.5 --detections $DETPKL --output-dir visualizations/HFV_FashionV2_val/$MODEL_NAME  --class_list_file /media/data/chnxi/HomeFurniture/taxonomy/furniture_fashion_visualintent_166_labels.txt


TESTSET=bing5k_fashion #FashionV2_val
MODEL_NAME=attr_frcnn_bs16_449
VIS_PATH=/home/chnxi/FashionV3/Models/attr_frcnn_bs16_449/visualizations/$TESTSET/
DETPKL=/home/chnxi/FashionV3/Models/attr_frcnn_bs16_449/attr_frcnn_bs16_449_bing5k/inference/model_final/detections.pkl

python tools/visualize_results.py --dataset $TESTSET --thresh 0.5 --detections $DETPKL --output-dir $VIS_PATH  --class_list_file /media/data/chnxi/FashionV2/taxonomy/fashion_intentbg_110_labels.txt
