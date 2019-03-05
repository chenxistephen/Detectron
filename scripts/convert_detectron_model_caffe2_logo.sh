MODEL_NAME=frcnn_scale600_roires14_sampleratio2_rpnratio2
python2 tools/convert_pkl_to_pb.py \
    --cfg configs/Logo/e2e_faster_rcnn_R-50-FPN_1x.yaml \
    --out_dir ./Trained_Models/Logo/$MODEL_NAME/ \
    TEST.WEIGHTS Trained_Models/Logo/$MODEL_NAME/model_final.pkl \

