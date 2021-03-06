
MODEL_NAME=$1
FOLDER=Logo

CUDA_VISIBLE_DEVICES=2 python2 tools/infer_detection_bbox.py \
    --cfg configs/Logo/logo_retinanet_1gpu_R-101-FPN_1x.yaml \
    --output-dir visualizations/$FOLDER/$MODEL_NAME \
    --image-ext jpg \
    --im_or_folder /home/stephenchen/data/Logo/Images/ \
    --im_list /home/stephenchen/data/Logo/ImageSets/test.txt \
    --wts /home/stephenchen/Detectron/Trained_Models/$FOLDER/$MODEL_NAME/model_final.pkl \
    #--cls_thrsh_file Trained_Models/$FOLDER/$MODEL_NAME/thumbnail_classwise_thresholds_fix_val_prec_0.9.csv
