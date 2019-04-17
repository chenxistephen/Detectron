import json
import os
import os.path as osp
import sys
from PIL import Image
import pickle as pkl
import cv2

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object
import detectron.utils.vis as vis_utils


modelType = 'COCO'
test_dataset = sys.argv[1] if len(sys.argv) > 1 else 'FashionV2_train' # {'HF', 'OpenImage'}


if modelType == 'HF_Detector':
    visPath = '/home/chnxi/Detectron/backfill/visualizations/HF_Detector/'
    cls_thrsh_file = '/home/chnxi/Detectron/backfill/HF_Detector/class_thresholds_at_pr.pkl'
    if test_dataset == 'FashionV2_train':
        bboxFile = '/home/chnxi/Detectron/backfill/HF_Detector/FashionV2_train/test/FashionV2_train/retinanet/detections.pkl' #sys.argv[1]
        imgPath = '/media/data/chnxi/FashionV2/Images/'
    elif test_dataset == 'OpenImage_train':
        bboxFile = '/home/chnxi/Detectron/backfill/HF_Detector/OpenImage_train/test/OpenImage_train/retinanet/detections.pkl' #sys.argv[1]
        imgPath = '/media/data/chnxi/OpenImage/train_images/'
    elif test_dataset == 'coco_2014_valminusminival':
        bboxFile = '/home/chnxi//Detectron/backfill/HF_Detector/coco_trainval/test/coco_2014_valminusminival/retinanet/detections.pkl' 
        imgPath = '/media/data/chnxi/coco/val2014/'
elif modelType == 'COCO':
    visPath = '/home/chnxi/Detectron/backfill/visualizations/COCOTrainval/'
    cls_thrsh_file = '/home/chnxi/Detectron/backfill/COCO_Mask_Detector/class_thresholds_at_pr.pkl'
    if test_dataset == 'FashionV2_train':
        bboxFile = '/home/chnxi/Detectron/backfill/COCO_Mask_Detector/FashionV2_train/test/FashionV2_train/generalized_rcnn/detections.pkl'
        imgPath = '/media/data/chnxi/FashionV2/Images/'
    elif test_dataset == 'OpenImage_train':
        bboxFile = '/home/chnxi/Detectron/backfill//COCO_Mask_Detector/OpenImage_train/test/OpenImage_train/generalized_rcnn/detections.pkl'
        imgPath = '/media/data/chnxi/OpenImage/train_images/'
    elif test_dataset == 'furniture_train':
        bboxFile = '/home/chnxi//Detectron/backfill/COCO_Mask_Detector/furniture_train/test/furniture_train/generalized_rcnn/detections.pkl'
        imgPath = '/media/data/chnxi/coco/val2014/'
    
   
visFolder = osp.join(visPath, test_dataset)
if not osp.isdir(visFolder):
    os.makedirs(visFolder)
    
print ("modelType = {}===================".format(modelType))
print ("test_dataset = {}===================".format(test_dataset))
print ("bboxFile = {}".format(bboxFile))
print ("imgPath = {}===================".format(imgPath))
print ("visFolder = {}===================".format(visFolder))

ds = JsonDataset(test_dataset)
roidb = ds.get_roidb()
    
print("loading {}".format(bboxFile))
dets = load_object(bboxFile)
print ("loaded")
all_boxes = dets['all_boxes']

cls_thrsh_dict = load_object(cls_thrsh_file)
print (cls_thrsh_dict.keys())
imgNum = len(roidb)
classes = cls_thrsh_dict['classes']

print(len(cls_thrsh_dict['prec_at'][0.9]))
print (len(classes))

print(roidb[0])
print (imgNum)

new_cls_thrsh_dict = {'prec_at_0.9': cls_thrsh_dict['prec_at'][0.9], 
                  'prec_at_0.8': cls_thrsh_dict['prec_at'][0.8],
                  'rec_at_0.5' : cls_thrsh_dict['rec_at'][0.5],
                  'rec_at_0.7' : cls_thrsh_dict['rec_at'][0.7]}
    
print (new_cls_thrsh_dict['prec_at_0.9'])

import random
visNum = 100
imgIds = random.sample(range(imgNum), visNum)
print(imgIds)

# print (new_cls_thrsh_dict['prec_at_0.9'])
import os
import os.path as osp
visPath = '../visualizations/'
if not osp.isdir(visPath):
    os.makedirs(visPath)
    
keylist = ['prec_at_0.9', 'prec_at_0.8', 'rec_at_0.5', 'rec_at_0.7']
color_list = ['g', 'r', 'b', 'm']

def visImageClsThrshs(plt, splot, im, imgBoxes, ecolor='m', title='', showCateg=False):
    #plt.cla()
    fx, fy, fi = splot
    plt.subplot(fx, fy, fi)
    plt.imshow(im)
    plt.axis('off')
    plt.title(title)
    ax = plt.gca()
    for cid, cls_boxes in enumerate(imgBoxes):
        for obj in cls_boxes:
            #print obj
            bbox = obj[:4]
            score = obj[-1]
            categ = classes[cid]            
            #ecolor = 'm' #if gid == jmax else gtcolor
            #print "{}: bbox = {}".format(categ, bbox)
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=ecolor, linestyle='-', linewidth=0.5))
            if showCateg:
                ax.text(bbox[0], bbox[3], '{}:{:.1f}'.format(categ, score), bbox={'facecolor': ecolor, 'alpha': 0.5})
        plt.draw()

import matplotlib.pyplot as plt
from PIL import Image
saveVis = False
showVis = True
        

    
def id_or_index(ix, val):
    if len(val) == 0:
        return val
    else:
        return val[ix]
        
for iid, ix in enumerate(imgIds): #ix, entry in enumerate(roidb):
    imgFileName = roidb[ix]['image']
    imgBaseName = osp.basename(imgFileName)
    print ("{}:{}".format(iid, imgBaseName))
    print (imgFileName)
    im = Image.open(imgFileName)
    img_boxes = [
        id_or_index(ix, cls_k_boxes) for cls_k_boxes in all_boxes
    ]
    img_boxes = img_boxes[1:] # remove background boxes
        
    f, axarr = plt.subplots(1, 4, sharey=True, figsize=(20, 20))
    plt.cla()
    fid = 0
    total_box_num = 0
    for fid, key in enumerate(keylist): #['prec_at_0.9']: #new_cls_thrsh_dict:
        print (key)
        ax = axarr[fid]
        thresholds = new_cls_thrsh_dict[key]
        img_boxes_thrshed_num = 0
        img_boxes_thrshed = []
        for cid, cls_boxes in enumerate(img_boxes):
            #print (cls_boxes.shape)
            cls_boxes_thrshed = [bbox for bbox in cls_boxes if bbox[-1] > thresholds[cid]]
            cls_boxes_num = len(cls_boxes_thrshed)
            img_boxes_thrshed.append(cls_boxes_thrshed)
            img_boxes_thrshed_num += cls_boxes_num
        print(img_boxes_thrshed_num)
        total_box_num += img_boxes_thrshed_num
        visImageClsThrshs(plt, (1,4, fid+1), im, img_boxes_thrshed, ecolor=color_list[fid], title=key, showCateg=True)
        fid += 1
    #plt.show()
    if total_box_num > 0:
        plt.savefig(osp.join(visFolder,imgBaseName), bbox_inches='tight',pad_inches = 0)
    plt.close()
print ("Done!")

