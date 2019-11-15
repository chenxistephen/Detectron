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

import numpy as np

def checkMkdir(dirname):
    if not osp.isdir(dirname):
        os.makedirs(dirname)
        
        
platform = '' # {'philly', 'aml', ''}


if platform == 'philly':
    imgPathDict =  {'COCO_trainval': None,
                    'coco_2014_train': 'coco/train2014/',
                    'coco_2014_valminusminival': 'coco/val2014/',
                    'furniture_train': 'HomeFurniture/Images.zip@/Images/',
                    'FashionV2_train': 'FashionV2/Images.zip@/Images/',
                    'OpenImage_train': 'OpenImage/train_images.zip@/train_images/',
                    'Object365_train': 'Object365/train.zip@/train/'
                    }
else:
    imgPathDict =  {'COCO_trainval': None,
                    'coco_2014_train': 'coco/train2014/',
                    'coco_2014_valminusminival': 'coco/val2014/',
                    'furniture_train': 'HomeFurniture/Images/',
                    'FashionV2_train': 'FashionV2/Images/',
                    'OpenImage_train': 'OpenImage/train_images/',
                    'Object365_train': 'Object365/train/'
                    }
        
sampleSizeMap = {'COCO_trainval': None,
                 'furniture_train': None,
                 'FashionV2_train': None,
                 'Object365_train': None,
                 'OpenImage_train': 800000
                }
        
boxFilePath = '/media/data/chnxi/GOD/threshold_bboxes/'
mapping_date_ver = '20191017' #'201909'
mapFilePath = '/media/data/chnxi/GOD/mappings/{}/backfill_src_nameid_to_GOD_id_mappings/'.format(mapping_date_ver)
labelFile = '/media/data/chnxi/GOD/taxonomy/GOD_taxonomy_{}.tsv'.format(mapping_date_ver) # 'GOD_taxonomy_V1.tsv'


modelSetMap = {'Fashion_Detector'  : 'fashion',
               'HF_Detector'       : 'furniture',
               'COCO_Mask_Detector': 'COCO',
               'OpenImage_Detector': 'Open Image'}

datasetList = ['Object365_train', 'furniture_train', 'COCO_trainval', 'FashionV2_train', 'OpenImage_train']
#############   Creating taxonomy   #############################################
import pandas as pd
god_data = pd.read_csv(labelFile, delimiter='\t')
god_name_id_map = dict(zip(god_data['GOD_v1_name'], god_data['GOD_v1_id']))
god_classes = list(god_data['GOD_v1_name'])
print ("len(god_classes) original = {}".format(len(god_classes)))
god_classes = ['__background__'] + god_classes


#############   Creating Data   ################################################# 
#####  Creating one json for 4 datasets
platformStr = '_' + platform if len(platform) > 0 else ''
openStr = 'Open{}k'.format(int(sampleSizeMap['OpenImage_train']/1000)) if sampleSizeMap['OpenImage_train'] is not None else 'All'
mergedDataset = 'GOD_{}_O365Backfilled_{}{}'.format(openStr, mapping_date_ver, platformStr) #'GOD_Open40k'
# totalSampleNum = 0
# for dataset in datasetList:
#     if sampleSizeMap[dataset] is not None:
#         totalSampleNum += sampleSizeMap[dataset]
# mergedDataset = '{}_{}'.format(mergedDataset, totalSampleNum) if totalSampleNum is not None else mergedDataset
print ("mergedDataset = {}".format(mergedDataset))
outAnnoFolder = '/media/data/chnxi/GOD/json_annotations/'
checkMkdir(outAnnoFolder)
outAnnoFile = osp.join(outAnnoFolder, '{}_train.json'.format(mergedDataset))
print ("outAnnoFile = {}".format(outAnnoFile))
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
totalImgCnt = -1
data['images'] = []
data['annotations'] = []
#################################################################################
def getAnno(image_id, bbox, categ_id, box_id, anno_src='gt'):
    image_id = int(image_id) # convert np.ndarray.int64 to int in case json.dump not serializable
    boxw = float(bbox[2]-bbox[0]) # convert np.ndarray.float32 to float in case json.dump not serializable
    boxh = float(bbox[3]-bbox[1])
    nbbox = [float(bbox[0]), float(bbox[1]), boxw, boxh]
    anno = {'segmentation':[],
            'area': boxw*boxh,
            'iscrowd':0,
            'image_id':image_id,
            'bbox':nbbox, # because COCO format [l, t, w,h] #float(r),float(b)],
            'category_id':categ_id,
            'id':box_id,
            'anno_src': anno_src
            }
    return anno
#################################################################################
def filterAnnoCondition(dataset, modelName, god_class_name=None):
    if dataset == 'furniture_train' and modelName in ['COCO_Mask_Detector', 'OpenImage_Detector']:
        if 'home_or_office_furnishing_or_decor' not in god_class_name:
            #print ("filtering {} from {} in {}".format(god_class_name, modelName, dataset))
            return True
    return False
#################################################################################
for dataset in datasetList:
#     outAnnoFile = osp.join(outAnnoFolder, '{}_annotations.json'.format(dataset))
#     visPath = '/media/data/chnxi/GOD/visualizations/'
#     visFolder = osp.join(visPath, dataset)
#     checkMkdir(visFolder)

    sampleSize = sampleSizeMap[dataset]
    print ("sampleSize = {}".format(sampleSize))
    imgPath = imgPathDict[dataset]
    print ("imgPath = {}".format(imgPath))
    if dataset == 'furniture_train':
        gtSetName = 'furniture'
        # Only backfills "dinnerware_serveware" from OpenImage/COCO;  No fashion backfills on HF
        mapFile = mapFilePath + 'HF_dataset_sources_ids.pkl'
        srcFiles = {#'Fashion_Detector'  : 'furniture_train.json',
                    'COCO_Mask_Detector': 'furniture_train.pkl',
                    'OpenImage_Detector': 'furniture_train.pkl'}
    elif dataset == 'FashionV2_train':
        gtSetName = 'fashion'
        #imgPath = 'FashionV2/Images/'
        mapFile = mapFilePath + 'Fashion_dataset_sources_ids.pkl'
        srcFiles = {'HF_Detector'       : 'FashionV2_train.pkl',
                    'COCO_Mask_Detector': 'FashionV2_train.pkl',
                    'OpenImage_Detector': 'FashionV2_train.pkl'}
    elif dataset == 'Object365_train':
        gtSetName = 'O365'
        #imgPath = 'OpenImage/train_images/'
        mapFile = mapFilePath + 'Object365_dataset_sources_ids.pkl'
        srcFiles = {'Fashion_Detector'  : 'Object365_train.json',
                    'HF_Detector'       : 'Object365_train.pkl',
                    'OpenImage_Detector': 'Object365_train.pkl'}
    elif dataset == 'OpenImage_train':
        gtSetName = 'Open Image'
        #imgPath = 'OpenImage/train_images/'
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
    if dataset == 'COCO_trainval':
        datasetNames = ('coco_2014_train','coco_2014_valminusminival')
        roidb = []
        for dsName in datasetNames:
            ds = JsonDataset(dsName)
            rdb = ds.get_roidb(gt=True)
            roidb = roidb + rdb
    else: #if dataset in ['furniture_train', 'FashionV2_train', 'OpenImage_train', 'Object365_train']:
        print ("creating json dataset {}".format(dataset))
        ds = JsonDataset(dataset)
        roidb = ds.get_roidb(gt=True)
    imgNum = len(roidb)
    print ('imgNum = {}'.format(imgNum))
    if sampleSize is not None:
        sampleSize = min(imgNum, sampleSize) 
    dbSampleNum = sampleSize if sampleSize is not None else imgNum
    print ("sampling {} images from roidb {}".format(dbSampleNum, dataset))
    #################################################################################

    ### Loading ROIDB   #############################################################
    print ("loading {}".format(mapFile))
    src_god_map = pkl.load(open(mapFile,'rb'))
    print (src_god_map.keys())
#     cocoMap = src_god_map['name_to_god_id']['COCO']
#     print (cocoMap)
#     cocoIdMap = src_god_map['id_map']['COCO']
#     print(cocoIdMap)
#     print (len(cocoMap))
    #################################################################################

    ### Loading model bboxes   ######################################################              
    print ("Loading model bboxes!")
    srcBoxes = {}
    for modelName in srcFiles:
        box_file_name = srcFiles[modelName]
        boxFileName = osp.join(boxFilePath, modelName, box_file_name)
        print ("Loading boxFileName = {}".format(boxFileName))
        if modelName in ['COCO_Mask_Detector', 'HF_Detector']:
            # format: all_boxes[totalImgCnt][class_id]
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
    ###################### Sampling Image index  ################################################
    if sampleSize is not None:
        sampleInds = np.random.choice(imgNum, sampleSize)
    else:
        sampleInds = list(range(imgNum))
        
    print ("len(sampleInds) = {} ==========".format(len(sampleInds)))
    
    ########### Getting bbox from 4 sources  ################################################

    for iix, imgid_in_dataset in enumerate(sampleInds):
        imgid_in_dataset = int(imgid_in_dataset)
        entry = roidb[imgid_in_dataset]
        totalImgCnt += 1 # start from -1 + 1 = 0
        ###############################################
        ## data['images']
        ###############################################
        file_name = entry['image']
        imgName = osp.basename(file_name)
        if dataset == 'COCO_trainval':
            if 'train' in imgName:
                imgPath = imgPathDict['coco_2014_train'] #'coco/train2014/'
            elif 'val' in imgName:
                imgPath = imgPathDict['coco_2014_valminusminival'] #'coco/val2014/'
        imgFileName = imgPath + imgName
        imgH = entry['height']
        imgW = entry['width']
        img = {'file_name': imgFileName,
               'height': entry['height'],
               'width': entry['width'],
               'id':totalImgCnt # bug: imgid
               }
        data['images'].append(img)
        
        #print (img)
        ###############################################
        ## Add GT Boxes
        ###############################################
        src_god_id_map = src_god_map['id_map'][gtSetName]
        src_name_godid_map = src_god_map['name_to_god_id'][gtSetName]
        visName = gtSetName + '_GT'
        anno_src = gtSetName + '_GT'
        gt_boxes = entry['boxes']
        gt_src_cls_ids = entry['gt_classes']
        for bid, box in enumerate(gt_boxes):
            src_class_id = gt_src_cls_ids[bid]
            if src_class_id in src_god_id_map:
                god_class_id = src_god_id_map[src_class_id]
                #print ("======{}:{} ==> GOD:{} {}".format(gtSetName, src_class_id, god_class_id, god_classes[god_class_id]))
                anno = getAnno(totalImgCnt, box[:4], god_class_id, totalBoxCnt, anno_src=anno_src)
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
            # print ("{} ==> {}".format(modelName, modelSetName))
            if modelName in ['COCO_Mask_Detector', 'HF_Detector']:
                # format: all_boxes[imgid_in_dataset][class_id]
                for cid, cls_boxes in enumerate(srcBoxes[modelName][imgid_in_dataset]):
                    src_class_id = cid + 1 # all_boxes starts from 0
                    if src_class_id in src_god_id_map and len(cls_boxes) > 0:
                        god_class_id = src_god_id_map[src_class_id]
                        #print ("========{}:{} ==> GOD:{}:{}".format(modelSetName, src_class_id, god_class_id, god_classes[god_class_id]))
                        for bbox in cls_boxes:
                            anno = getAnno(totalImgCnt, bbox, god_class_id, totalBoxCnt, anno_src=modelName)
                            if filterAnnoCondition(dataset, modelName, god_class_name=god_classes[god_class_id]):
                                continue
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
                        if filterAnnoCondition(dataset, modelName, god_class_name=god_classes[god_class_id]):
                            continue
                        anno = getAnno(totalImgCnt, bbox, god_class_id, totalBoxCnt, anno_src=modelName)
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
                            anno = getAnno(totalImgCnt, bbox, god_class_id, totalBoxCnt, anno_src=modelName)
                            #print (anno)
                            data['annotations'].append(anno)
                            totalBoxCnt += 1
        if (iix + 1) % 100 == 0:
            print ("{}/{}".format(iix + 1, len(sampleInds)))
            print ("img = {}".format(img))
            anno = data['annotations'][-1]
            cat_id = anno['category_id']
            cat_name = data['categories'][cat_id-1]['name']
            print ("{}:{}, cat_id:name = {}::{}, (x,y,w,h) = {}".format(totalImgCnt, imgName, cat_id, cat_name, anno['bbox']))
#################################################################################
print ("Saving annotatinos to {}".format(outAnnoFile))
with open(outAnnoFile, 'w') as fout:
    json.dump(data, fout)
print ("Done")
#################################################################################
### Double check the anno instance source dataset
from collections import Counter
img_id_names = {img['id']: img['file_name'] for img in data['images'] }
imgNames2 = [img_id_names[a['image_id']].split('/')[0] for a in all_anno['annotations']]
CA = Counter(imgNames2)
print (CA)
               
               
               
               
               
               
               