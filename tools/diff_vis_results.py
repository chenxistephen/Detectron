#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Script for visualizing results saved in a detections.pkl file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import os
import sys

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object
import detectron.utils.vis as vis_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset',
        default='coco_2014_minival',
        type=str
    )
    parser.add_argument(
        '--detections1',
        dest='detections1',
        help='detections pkl file 1',
        default='',
        type=str
    )
    parser.add_argument(
        '--detections2',
        dest='detections2',
        help='detections pkl file 2',
        default='',
        type=str
    )
    parser.add_argument(
        '--class_list_file1',
        dest='class_list_file1',
        help='class_list_file1',
        default=None,
        type=str
    )
    parser.add_argument(
        '--class_list_file2',
        dest='class_list_file2',
        help='class_list_file2',
        default=None,
        type=str
    )
    parser.add_argument(
        '--cls_thrsh_file1',
        dest='cls_thrsh_file1',
        help='cls_thrsh_file1',
        default=None,
        type=str
    )
    parser.add_argument(
        '--cls_thrsh_file2',
        dest='cls_thrsh_file2',
        help='cls_thrsh_file2',
        default=None,
        type=str
    )
    parser.add_argument(
        '--thresh1',
        dest='thresh1',
        help='detection prob thresh1old',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--thresh2',
        dest='thresh2',
        help='detection prob thresh1old',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='./tmp/vis-output',
        type=str
    )
    parser.add_argument(
        '--first',
        dest='first',
        help='only visualize the first k images',
        default=0,
        type=int
    )
    parser.add_argument(
        '--sampleNum',
        dest='sampleNum',
        help='random sample image num',
        default=None,
        type=int,
    )
#     parser.add_argument(
#         '--range',
#         dest='range',
#         help='start (inclusive) and end (exclusive) indices',
#         default=None,
#         type=int,
#         nargs=2
#     )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

import numpy as np
def vis(dataset, detections_pkl1, detections_pkl2, output_dir,  thresh1=0.5, thresh2=0.5, cls_thrsh_file1=None, cls_thrsh_file2=None, sampleNum=None, class_list_file1=None, class_list_file2=None):
    ds = JsonDataset(dataset)
    #classes_list = [l.rstrip() for l in open('/home/chnxi/data/HomeFurniture/taxonomy/furniture_58_labels.txt','r').readlines()]
    if class_list_file1 is not None:
        classes_list = [l.rstrip().split('\t')[0].split('\\')[-1] for l in open(class_list_file1,'r').readlines()]
    else:
        classes_list = ds.classes
    classes_list = ['background'] + classes_list
    print (classes_list)
    
    if class_list_file2 is not None:
        classes_list2 = [l.rstrip().split('\t')[0].split('\\')[-1] for l in open(class_list_file2,'r').readlines()]
    else:
        classes_list2 = ds.classes
    classes_list2 = ['background'] + classes_list2
    print (classes_list2)
    
    
    roidb = ds.get_roidb()
    
    if 'range' in detections_pkl1:
        pkl_range = osp.splitext(s)[0].split('range_')[-1].split('_')
        pkl_range = [int(r) for r in id_range]
    else:
        pkl_range = [0, len(roidb)]
        
    pkl_img_ids = list(range(pkl_range[0], pkl_range[1]))
    pklNum = len(pkl_img_ids)
    
    if sampleNum is not None:
        ids_in_pkl = np.random.choice(pklNum, sampleNum)
    else:
        ids_in_pkl = list(range(pklNum))
        
    dets1 = load_object(detections_pkl1)
    dets2 = load_object(detections_pkl2)
    
    if cls_thrsh_file1 is not None:
        clsPR1 = load_object(cls_thrsh_file1)
        cls_thrsh_list1 = clsPR1['cls_thrsh_at_prec'][0.9]
    else:
        cls_thrsh_list1 = None
    if cls_thrsh_file2 is not None:
        clsPR2 = load_object(cls_thrsh_file2)    
        cls_thrsh_list2 = clsPR2['cls_thrsh_at_prec'][0.9]
    else:
        cls_thrsh_list2 = None
    
    print ("pkl_range = {}, sampleNum = {}, len(ids_in_pkl) = {}, len(roidb) = {}".format(pkl_range, sampleNum, len(ids_in_pkl), len(roidb)))

    assert all(k in dets1 for k in ['all_boxes', 'all_segms', 'all_keyps']), \
        'Expected detections pkl file in the format used by test_engine.py'
    
    assert all(k in dets2 for k in ['all_boxes', 'all_segms', 'all_keyps']), \
        'Expected detections pkl file in the format used by test_engine.py'

    all_boxes1 = dets1['all_boxes']
    all_boxes2 = dets2['all_boxes']


    def id_or_index(ix, val):
        if len(val) == 0:
            return val
        else:
            return val[ix]
        
        
    #for ix, entry in enumerate(roidb):
    for ii, id_in_pkl in enumerate(ids_in_pkl):
        img_id = pkl_img_ids[id_in_pkl]
        entry = roidb[img_id]
        if ii % 10 == 0:
            print('{:d}/{:d}'.format(ii + 1, len(ids_in_pkl)))

        im = cv2.imread(entry['image'])
        im_name = os.path.splitext(os.path.basename(entry['image']))[0]

        img_cls_boxes_1 = [
            id_or_index(id_in_pkl, cls_k_boxes) for cls_k_boxes in all_boxes1[:len(classes_list)]
        ]
    
        img_cls_boxes_2 = [
            id_or_index(id_in_pkl, cls_k_boxes) for cls_k_boxes in all_boxes2[:len(classes_list2)]
        ]
        vis_utils.diff_vis_one_image(
            im[:, :, ::-1],
            '{:d}_{:s}'.format(img_id, im_name),
            os.path.join(output_dir, 'vis_thrsh_P90'),
            img_cls_boxes_1,
            img_cls_boxes_2,
            thresh1=thresh1,
            thresh2=thresh2,
            box_alpha=0.8,
            dataset=ds,
            show_class=True, 
            ext='.png', 
            classes_list=classes_list,
            classes_list2=classes_list2,
            cls_thrsh_list1=cls_thrsh_list1,
            cls_thrsh_list2=cls_thrsh_list2,
            out_when_no_box=False
        )


if __name__ == '__main__':
    opts = parse_args()
    vis(
        opts.dataset,
        opts.detections1,
        opts.detections2,
        opts.output_dir,
        thresh1=opts.thresh1,
        thresh2=opts.thresh2,
        sampleNum=opts.sampleNum,
        class_list_file1=opts.class_list_file1,
        class_list_file2=opts.class_list_file2,
        cls_thrsh_file1=opts.cls_thrsh_file1,
        cls_thrsh_file2=opts.cls_thrsh_file2
    )
