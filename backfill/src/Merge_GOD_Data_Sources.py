import json
import os
import os.path as osp
import sys
from PIL import Image
import pickle as pkl
import cv2

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object, save_object
import detectron.utils.vis as vis_utils

def checkMkdir(dirname):
    if not osp.isdir(dirname):
        os.makedirs(dirname)
        
boxFilePath = '/media/data/chnxi/GOD/threshold_bboxes/'
mapFilePath = '/media/data/chnxi/GOD/mappings/src_nameid_to_GOD_id_mappings/'

orgAnnoFile = '/media/data/chnxi/HomeFurniture/json_annotations/train_annotations.json'
modelSetMap = {'Fashion_Detector'  : 'fashion',
               'HF_Detector'       : 'furniture',
               'COCO_Mask_Detector': 'COCO',
               'OpenImage_Detector': 'Open Image'}

datasetList = ['COCO_trainval', 'furniture_train', 'FashionV2_train',  'OpenImage_train']
#############   Creating taxonomy   #############################################
import pandas as pd
labelFile = '/media/data/chnxi/GOD/taxonomy/GOD_taxonomy_V1.tsv'
god_data = pd.read_csv(labelFile, delimiter='\t')
god_name_id_map = dict(zip(god_data['GOD_v1_name'], god_data['GOD_v1_id']))
god_classes = list(god_data['GOD_v1_name'])
print ("len(god_classes) original = {}".format(len(god_classes)))
god_classes = ['__background__'] + god_classes


#############   Creating Data   ################################################# 
#####  Creating one json for 4 datasets
mergedDataset = 'GOD'
outAnnoFolder = '/media/data/chnxi/GOD/json_annotatinos/'
checkMkdir(outAnnoFolder)
outAnnoFile = osp.join(outAnnoFolder, '{}_train_annotations.json'.format(mergedDataset))
data = {}
supercat = 'Generic'
data['info'] = {'description':'Bing Generic Object Detection dataset on {}'.format(mergedDataset)}
data['licenses'] = [{}]
data['type'] = 'instances'

data['categories'] = []

for cls, id in god_name_id_map.items():
    cats = {'supercategory':supercat,
            'id':id,
            'name':cls}
    data['categories'].append(cats)

totalBoxCnt = 0
data['images'] = []
data['annotations'] = []
#################################################################################

for dataset in datasetList:
#     outAnnoFile = osp.join(outAnnoFolder, '{}_annotations.json'.format(dataset))
#     visPath = '/media/data/chnxi/GOD/visualizations/'
#     visFolder = osp.join(visPath, dataset)
#     checkMkdir(visFolder)

    if dataset == 'furniture_train':
        gtSetName = 'furniture'
        imgPath = 'HomeFurniture/Images/'
        mapFile = mapFilePath + 'HF_dataset_sources_ids.pkl'
        srcFiles = {'Fashion_Detector'  : 'furniture_train.json',
                    'COCO_Mask_Detector': 'furniture_train.pkl',
                    'OpenImage_Detector': 'furniture_train.pkl'}
    elif dataset == 'FashionV2_train':
        gtSetName = 'fashion'
        imgPath = 'FashionV2/Images/'
        mapFile = mapFilePath + 'Fashion_dataset_sources_ids.pkl'
        srcFiles = {'HF_Detector'       : 'FashionV2_train.pkl',
                    'COCO_Mask_Detector': 'FashionV2_train.pkl',
                    'OpenImage_Detector': 'FashionV2_train.pkl'}
    elif dataset == 'OpenImage_train':
        gtSetName = 'Open Image'
        imgPath = 'OpenImage/train_images.zip@/train_images/'
        mapFile = mapFilePath + 'OpenImage_dataset_sources_ids.pkl'
        srcFiles = {'Fashion_Detector'  : 'OpenImage_train.json',
                    'HF_Detector'       : 'OpenImage_train.pkl',
                    'COCO_Mask_Detector': 'OpenImage_train.pkl'}
    elif dataset == 'COCO_trainval':
        gtSetName = 'COCO'
        mapFile = mapFilePath + 'COCO_dataset_sources_ids.pkl'
        srcFiles = {'Fashion_Detector'  : 'COCO_trainval.json',
                    'HF_Detector'       : 'coco_2014_trainval.pkl',
                    'OpenImage_Detector': 'COCO_trainval.pkl'}


    #################################################################################
    ### Loading ROIDB   #############################################################

    if dataset in ['furniture_train', 'FashionV2_train', 'OpenImage_train']:
        print ("creating json dataset {}".format(dataset))
        ds = JsonDataset(dataset)
        roidb = ds.get_roidb(gt=True)
    elif dataset == 'COCO_trainval':
        datasetNames = ('coco_2014_train','coco_2014_valminusminival')
        roidb = []
        for dsName in datasetNames:
            ds = JsonDataset(dsName)
            rdb = ds.get_roidb(gt=True)
            roidb = roidb + rdb
    imgNum = len(roidb)
    print ('imgNum = {}'.format(imgNum))
    #################################################################################

    ### Loading ROIDB   #############################################################
    print ("loading {}".format(mapFile))
    src_god_map = pkl.load(open(mapFile,'rb'))
    print (src_god_map.keys())
    cocoMap = src_god_map['name_to_god_id']['COCO']
    print (cocoMap)
    cocoIdMap = src_god_map['id_map']['COCO']
    print(cocoIdMap)
    print (len(cocoMap))
    #################################################################################

    ### Loading model bboxes   ######################################################              
    print ("Loading model bboxes!")
    srcBoxes = {}
    for modelName in srcFiles:
        box_file_name = srcFiles[modelName]
        boxFileName = osp.join(boxFilePath, modelName, box_file_name)
        print ("Loading boxFileName = {}".format(boxFileName))
        if modelName in ['COCO_Mask_Detector', 'HF_Detector']:
            # format: all_boxes[imgid][class_id]
            all_boxes = load_object(boxFileName)
            print ("loaded")
            print ("all_boxes.size = ({}, {})".format(len(all_boxes), len(all_boxes[0])))
            srcBoxes[modelName] = all_boxes
        elif modelName == 'OpenImage_Detector':
            # all_boxes[file_name] = {'file_name': file_name,
            #        'boxes': boxes,
            #        'label_names': label_names,
            #        'labels': labels}
            all_boxes = load_object(boxFileName)
            print ("len(all_boxes) = {}".format(len(all_boxes)))
            srcBoxes[modelName] = all_boxes
        elif '.json' in box_file_name:
            # {'image_id': 'd7a0b91d2d83d457', 
            #'boxes': [{'score': 0.228, 'category_id': 2, 
            #'bbox': [506.49920000000003, 414.4128, 657.8248, 544.1536], 
            #'category_name': 'accessories-glasses'}], 'image_file_name': 'd7a0b91d2d83d457.jpg'}
            print ("loading {}".format(boxFileName))
            dets = json.load(open(boxFileName,'r'))
            all_boxes = {}
            for ix, det in enumerate(dets):
                # make it a dictionary with image file name as key
                img_file_name = det['image_file_name']
                all_boxes[img_file_name] = det['boxes']
            print ("len(all_boxes) = {}".format(len(all_boxes)))
            print ("loaded")
            srcBoxes[modelName] = all_boxes #dets
        print ("srcBoxes[{}] loaded".format(modelName))
    #################################################################################

    def getAnno(image_id, bbox, categ_id, box_id):
        boxw = float(bbox[2]-bbox[0]) # convert np.ndarray.float32 to float in case json.dump not serializable
        boxh = float(bbox[3]-bbox[1])
        nbbox = [float(bbox[0]), float(bbox[1]), boxw, boxh]
        anno = {'segmentation':[],
                'area': boxw*boxh,
                'iscrowd':0,
                'image_id':image_id,
                'bbox':nbbox, # because COCO format [l, t, w,h] #float(r),float(b)],
                'category_id':categ_id,
                'id':box_id
                }
        return anno


    ########### Getting bbox from 4 sources  ################################################


    for imgid, entry in enumerate(roidb):
        # entry = roidb[imgid]
        ###############################################
        ## data['images']
        ###############################################
        file_name = entry['image']
        imgName = osp.basename(file_name)
        if dataset == 'COCO_trainval':
            if 'train' in imgName:
                imgPath = 'coco/train2014/'
            elif 'val' in imgName:
                imgPath = 'coco/val2014/'
        imgFileName = imgPath + imgName
        imgH = entry['height']
        imgW = entry['width']
        img = {'file_name': imgFileName,
               'height': entry['height'],
               'width': entry['width'],
               'id':imgid
               }
        data['images'].append(img)
        #print (img)
        ###############################################
        ## Add GT Boxes
        ###############################################
        src_god_id_map = src_god_map['id_map'][gtSetName]
        src_name_godid_map = src_god_map['name_to_god_id'][gtSetName]
        visName = gtSetName + '_GT'
        gt_boxes = entry['boxes']
        gt_src_cls_ids = entry['gt_classes']
        for bid, box in enumerate(gt_boxes):
            src_class_id = gt_src_cls_ids[bid]
            if src_class_id in src_god_id_map:
                god_class_id = src_god_id_map[src_class_id]
                #print ("======{}:{} ==> GOD:{} {}".format(gtSetName, src_class_id, god_class_id, god_classes[god_class_id]))
                anno = getAnno(imgid, box[:4], god_class_id, totalBoxCnt)
                #print (anno)
                data['annotations'].append(anno)
                totalBoxCnt += 1
        ###############################################
        ## Add boxes from other models
        ###############################################
        for modelName in srcFiles:
            modelSetName = modelSetMap[modelName]
            src_god_id_map = src_god_map['id_map'][modelSetName]
            src_name_godid_map = src_god_map['name_to_god_id'][modelSetName]
    #         print ("{} ==> {}".format(modelName, modelSetName))
            if modelName in ['COCO_Mask_Detector', 'HF_Detector']:
                # format: all_boxes[imgid][class_id]
                for cid, cls_boxes in enumerate(srcBoxes[modelName][imgid]):
                    src_class_id = cid + 1 # all_boxes starts from 0
                    if src_class_id in src_god_id_map and len(cls_boxes) > 0:
                        god_class_id = src_god_id_map[src_class_id]
                        #print ("========{}:{} ==> GOD:{}:{}".format(modelSetName, src_class_id, god_class_id, god_classes[god_class_id]))
                        for bbox in cls_boxes:
                            anno = getAnno(imgid, bbox, god_class_id, totalBoxCnt)
                            #if totalBoxCnt % 10 == 0:
                            #print (anno)
                            data['annotations'].append(anno)
                            totalBoxCnt += 1
            if modelName == 'OpenImage_Detector':
                # all_boxes[file_name] = {'file_name': file_name,
                #        'boxes': boxes,
                #        'label_names': label_names,
                #        'labels': labels}
                boxes = srcBoxes[modelName][imgName]['boxes']
                label_names = srcBoxes[modelName][imgName]['label_names']
                for bid, box in enumerate(boxes):
                    src_class_name = label_names[bid] # do not use labels: it's not correct for OpenImage Id
                    if src_class_name in src_name_godid_map:
                        god_class_id = src_name_godid_map[src_class_name]
                        #print ("========= {}:{} ==> GOD:{}:{}".format(modelSetName, src_class_name, god_class_id, god_classes[god_class_id]))
                        bbox = [box[0] * imgW, box[1] * imgH, box[2] * imgW, box[3] * imgH]
                        anno = getAnno(imgid, bbox, god_class_id, totalBoxCnt)
                        #print (anno)
                        data['annotations'].append(anno)
                        totalBoxCnt += 1                 
            if modelName == 'Fashion_Detector': #'.json' in box_file_name:
                # all_boxes[imgName] = [ {[506.49920000000003, 414.4128, 657.8248, 544.1536], 
                #'category_name': 'accessories-glasses'} ]
                if imgName in srcBoxes[modelName]:
                    boxes = srcBoxes[modelName][imgName]
                    for bid, obj in enumerate(boxes):
                        src_class_name = obj['category_name']
                        if src_class_name in src_name_godid_map:
                            #print (obj)
                            god_class_id = src_name_godid_map[src_class_name]
                            #print ("========{}:{} ==> GOD:{}{}".format(modelSetName, src_class_name, god_class_id, god_classes[god_class_id]))
                            bbox = obj['bbox']
                            anno = getAnno(imgid, bbox, god_class_id, totalBoxCnt)
                            #print (anno)
                            data['annotations'].append(anno)
                            totalBoxCnt += 1
        if (imgid + 1) % 100 == 0:
            print ("{}/{}".format(imgid + 1, len(roidb)))
            print ("img = {}".format(img))
            anno = data['annotations'][-1]
            print ("{}: cat_id = {}, (x,y,w,h) = {}".format(imgName, anno['category_id'], anno['bbox']))
#################################################################################
print ("Saving annotatinos to {}".format(outAnnoFile))   
with open(outAnnoFile, 'w') as fout:
    json.dump(data, fout)
print ("Done")
#################################################################################
    
               
               
               
               
               
               
               