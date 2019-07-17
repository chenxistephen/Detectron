#!/bin/bash
set -eu


MODELPATH=/media/data/chnxi/FashionV3/Models/

# MODEL_NAME=hf_fashion_vi_nd

#DETPKL=/media/data/chnxi/FashionV3/Models/hf_fashion_vi_nd/Train_400-Test_400-SoftNMS-True/HFV_FashionV2_val/retinanet/detections.pkl

# python tools/visualize_results.py --dataset HFV_FashionV2_val --thresh 0.5 --detections $DETPKL --output-dir visualizations/FashionV3/$MODEL_NAME  --class_list_file /media/data/chnxi/HomeFurniture/taxonomy/furniture_fashion_visualintent_166_labels.txt

#DETPKL=/home/chnxi/FashionV3/Models/hf_fashion_vi_nd/400/detections.pkl

#DETPKL=/home/chnxi/FashionV3/Models/hf_fashion_vi_nd/Train_400-Test_400-SoftNMS-True/furniture_val/retinanet/detections.pkl

#python tools/visualize_results.py --dataset furniture_val --thresh 0.5 --detections $DETPKL --output-dir visualizations/HFV2/$MODEL_NAME  --class_list_file /media/data/chnxi/HomeFurniture/taxonomy/furniture_fashion_visualintent_166_labels.txt



DETPKL1=/media/data/chnxi/FashionV3/Models/fashionv3_retr50_iter360000_lr0.005_alpha0.5_gamma2.0/Test_600-SoftNMS-True/FashionV2_val/retinanet/detections.pkl

CTH1=/media/data/chnxi/FashionV3/Models/fashionv3_retr50_iter360000_lr0.005_alpha0.5_gamma2.0/Test_600-SoftNMS-True/FashionV2_val/retinanet/classwise_pr_curves.pkl

DETPKL2=/media/data/chnxi/FashionV3/Models/Fashion_HF_VI/gn_v0/Test_600-SoftNMS-True/FashionV2_val/retinanet/detections.pkl

CTH2=/media/data/chnxi/FashionV3/Models/Fashion_HF_VI/gn_v0/Test_600-SoftNMS-True/FashionV2_val/retinanet/classwise_pr_curves.pkl


DATASET=FashionV2_val

python tools/diff_vis_results.py --dataset $DATASET --thresh 0.5 --detections1 $DETPKL1 --detections2 $DETPKL2 --cls_thrsh_file1 $CTH1  --cls_thrsh_file2 $CTH2 --output-dir /media/data/chnxi/FashionV3/visualizations/diff_FashionHFVI_FashionVI/$DATASET/Test600  --class_list_file /media/data/chnxi/FashionV2/taxonomy/fashion_82_labels.txt
