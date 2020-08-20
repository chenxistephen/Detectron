#!/bin/bash
set -eu

GPUS=1

MODEL_PATH=/media/data/chnxi/GOD/Models

#MODEL_NAME=FashionVI-GN_400_8gpu_iter_180000-lr_0.01-alpha_0.5-gamma_2.

MODEL_NAME=Final_Candidate #Open800k_580/retr50_bs_4-iter_450000-lr_0.01
#god_frcnn_tuneiter_1000000_nonccl_try7

SOFTNMS=False
SCALE=400

echo $MODEL_NAME
echo "Test Scale =" $SCALE

CUDA_VISIBLE_DEVICES="$GPUS" python tools/convert_pkl_to_pb.py \
        --cfg configs/GOD/retinanet_R-50-FPN_8gpu.yaml \
        --net_name GODV1_ret50 \
        --out_dir $MODEL_PATH/$MODEL_NAME/caffe2_pb_model \
        --device cpu \
        --fuse_af 0 \
        TEST.WEIGHTS $MODEL_PATH/$MODEL_NAME/model_final.pkl \
        #TEST.SOFT_NMS.ENABLED $SOFTNMS \
        #TEST.SCALE $SCALE \
        #--fuse_af 0 \
        #        --test_img /media/data/chnxi/FashionV2/Images/f8073fd86ca4ec881f08ddbc38623662.jpg \