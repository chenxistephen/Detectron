#!/bin/bash
set -eu


MODELPATH=/media/data/chnxi/FashionV3/Models/

# MODEL_NAME=hf_fashion_vi_nd

#DETPKL=/media/data/chnxi/FashionV3/Models/hf_fashion_vi_nd/Train_400-Test_400-SoftNMS-True/HFV_FashionV2_val/retinanet/detections.pkl

# python tools/visualize_results.py --dataset HFV_FashionV2_val --thresh 0.5 --detections $DETPKL --output-dir visualizations/FashionV3/$MODEL_NAME  --class_list_file /media/data/chnxi/HomeFurniture/taxonomy/furniture_fashion_visualintent_166_labels.txt

#DETPKL=/home/chnxi/FashionV3/Models/hf_fashion_vi_nd/400/detections.pkl

#DETPKL=/home/chnxi/FashionV3/Models/hf_fashion_vi_nd/Train_400-Test_400-SoftNMS-True/furniture_val/retinanet/detections.pkl

#python tools/visualize_results.py --dataset furniture_val --thresh 0.5 --detections $DETPKL --output-dir visualizations/HFV2/$MODEL_NAME  --class_list_file /media/data/chnxi/HomeFurniture/taxonomy/furniture_fashion_visualintent_166_labels.txt

###############################################################################################
# MODELPATH1=/media/data/chnxi/FashionV3/Models/fashionv3_retr50_iter360000_lr0.005_alpha0.5_gamma2.0/Test_400-SoftNMS-True/FashionV2_val/retinanet

# DETPKL1=$MODELPATH1/detections.pkl
# CTH1=$MODELPATH1/classwise_pr_curves.pkl
###############################################################################################
###############################################################################################
#DATASET=FashionV2_val
DATASET=bing5k_fashion
###############################################################################################
MODELPATH1=/media/data/chnxi/FashionV3/Models/FashionVI-GN_400_8gpu_iter_180000-lr_0.01-alpha_0.5-gamma_2.0/Test_400-SoftNMS-True/$DATASET/retinanet

DETPKL1=$MODELPATH1/detections.pkl
CTH1=/media/data/chnxi/FashionV3/Models/FashionVI-GN_400_8gpu_iter_180000-lr_0.01-alpha_0.5-gamma_2.0/Test_400-SoftNMS-True/FashionV2_val/retinanet/classwise_pr_curves.pkl
###############################################################################################

MODELPATH2=/media/data/chnxi/FashionV3/Models/FHV-GN_400_8gpu_iter_300000-lr_0.01-alpha_0.25-gamma_2.0/Test_400-SoftNMS-True/$DATASET/retinanet

DETPKL2=$MODELPATH2/detections.pkl
CTH2=/media/data/chnxi/FashionV3/Models/FHV-GN_400_8gpu_iter_300000-lr_0.01-alpha_0.25-gamma_2.0/Test_400-SoftNMS-True/FashionV2_val/retinanet/classwise_pr_curves.pkl
###############################################################################################


python tools/diff_vis_results.py --dataset $DATASET --thresh1 0.557 --thresh2 0.425 --detections1 $DETPKL1 --detections2 $DETPKL2 --output-dir /media/data/chnxi/FashionV3/visualizations/diff_FV400GN_FHV400GN/$DATASET/Test400  --class_list_file1 /media/data/chnxi/FashionV2/taxonomy/fashion_82_labels.txt --class_list_file2 /media/data/chnxi/FashionV2/taxonomy/fashion_82_labels.txt

#--cls_thrsh_file1 $CTH1  --cls_thrsh_file2 $CTH2
