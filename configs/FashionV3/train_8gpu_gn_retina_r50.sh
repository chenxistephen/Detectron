#!/bin/bash
set -eu
GPUS=$1
ITER=$2
LR=$3 #0.01
LOSS_ALPHA=$4 #0.25 #0.5 #0.25
LOSS_GAMMA=2.0
TRAINSET=Fashion_VI_train #Fashion_VI_train
CLSNUM=111 #167

# PREMODEL=/data/chnxi/Detectron/detectron-output/FHV-GN_400_8gpu_iter_300000-lr_0.01-alpha_0.25-gamma_2.0/train/Fashion_HF_VI_train/retinanet/model_final.pkl

#PREMODEL=https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl



CUDA_VISIBLE_DEVICES="$GPUS" python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/FashionV3/retinanet_R-50-FPN_2x_gn_8gpu.yaml \
    NUM_GPUS 8 \
    MODEL.NUM_CLASSES $CLSNUM \
    SOLVER.BASE_LR $LR \
    SOLVER.MAX_ITER $ITER \
    TRAIN.DATASETS "('$TRAINSET', )" \
    RETINANET.LOSS_ALPHA $LOSS_ALPHA \
    RETINANET.LOSS_GAMMA $LOSS_GAMMA \
    OUTPUT_DIR ./detectron-output/FashionVI-GN_400_8gpu_iter_$ITER-lr_$LR-alpha_$LOSS_ALPHA-gamma_$LOSS_GAMMA \
    USE_NCCL True | tee logs/FashionV3/FashionVI-GN_400_8gpu_iter_$ITER-lr_$LR-alpha_$LOSS_ALPHA-gamma_$LOSS_GAMMA.log
    #     TRAIN.WEIGHTS $PREMODEL \
