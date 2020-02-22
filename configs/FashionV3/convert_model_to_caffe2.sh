#!/bin/bash
set -eu

GPUS=1

MODEL_PATH=/media/data/chnxi/FashionV3/Models

#MODEL_NAME=FashionVI-GN_400_8gpu_iter_180000-lr_0.01-alpha_0.5-gamma_2.

MODEL_NAME=Fashion_HF_VI/bs2-iter_300000-lr_0.01-alpha_0.25-gamma_2
#MODEL_NAME=fashionv3_retr50_iter360000_lr0.005_alpha0.5_gamma2.0

SOFTNMS=False
SCALE=400

echo $MODEL_NAME
echo "Test Scale =" $SCALE

CUDA_VISIBLE_DEVICES="$GPUS" python tools/convert_pkl_to_pb.py \
        --cfg configs/FashionV3/fashion_vi_110_retinanet.yaml \
        --net_name FashionV3_FashionVIGN \
        --out_dir $MODEL_PATH/$MODEL_NAME/caffe2_pb_model \
        --device cpu \
        TEST.WEIGHTS $MODEL_PATH/$MODEL_NAME/model_final.pkl \
        #TEST.SOFT_NMS.ENABLED $SOFTNMS \
        #TEST.SCALE $SCALE \
        #        --test_img /media/data/chnxi/FashionV2/Images/f8073fd86ca4ec881f08ddbc38623662.jpg \