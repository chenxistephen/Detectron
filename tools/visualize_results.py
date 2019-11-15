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
        '--detections',
        dest='detections',
        help='detections pkl file',
        default='',
        type=str
    )
    parser.add_argument(
        '--class_list_file',
        dest='class_list_file',
        help='class_list_file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.7,
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
def vis(dataset, detections_pkl, thresh, output_dir, sampleNum=None, class_list_file=None, rm_prev=True):
    if rm_prev:
        command = 'rm -r {}'.format(output_dir)
        print (command)
        os.system(command)
    ds = JsonDataset(dataset)
    #classes_list = [l.rstrip() for l in open('/home/chnxi/data/HomeFurniture/taxonomy/furniture_58_labels.txt','r').readlines()]
    #classes_list = [l.rstrip().split('\t')[0].split('\\')[-1] for l in open(class_list_file,'r').readlines()]
    if class_list_file is not None:
        classes_list = [l.rstrip().split('\t')[1].split('/')[-1] for l in open(class_list_file,'r').readlines()[1:]]
    else:
        classes_list = ds.classes
    classes_list = ['background'] + classes_list
    
    print (classes_list)
    roidb = ds.get_roidb()
    
    if 'range' in detections_pkl:
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
        
    dets = load_object(detections_pkl)
    
    print ("pkl_range = {}, sampleNum = {}, len(ids_in_pkl) = {}, len(roidb) = {}".format(pkl_range, sampleNum, len(ids_in_pkl), len(roidb)))

    assert all(k in dets for k in ['all_boxes', 'all_segms', 'all_keyps']), \
        'Expected detections pkl file in the format used by test_engine.py'

    all_boxes = dets['all_boxes']
    all_segms = dets['all_segms']
    all_keyps = dets['all_keyps']

    def id_or_index(ix, val):
        if len(val) == 0:
            return val
        else:
            return val[ix]
        
        
    #for ix, entry in enumerate(roidb):
    for ii, id_in_pkl in enumerate(ids_in_pkl):
        if ii % 10 == 0:
            print('{:d}/{:d}'.format(ii + 1, len(ids_in_pkl)))
        img_id = pkl_img_ids[id_in_pkl]
        entry = roidb[img_id]
        im = cv2.imread(entry['image'])
        im_name = os.path.splitext(os.path.basename(entry['image']))[0]

        cls_boxes_i = [
            id_or_index(id_in_pkl, cls_k_boxes) for cls_k_boxes in all_boxes
        ]
        cls_segms_i = [
            id_or_index(id_in_pkl, cls_k_segms) for cls_k_segms in all_segms
        ]
        cls_keyps_i = [
            id_or_index(id_in_pkl, cls_k_keyps) for cls_k_keyps in all_keyps
        ]

        vis_utils.vis_one_image(
            im[:, :, ::-1],
            '{:d}_{:s}'.format(img_id, im_name),
            os.path.join(output_dir, 'vis_thrsh_{}'.format(thresh)),
            cls_boxes_i,
            segms=cls_segms_i,
            keypoints=cls_keyps_i,
            thresh=thresh,
            box_alpha=0.8,
            dataset=ds,
            show_class=True, 
            ext='.png', 
            classes_list=classes_list
        )


if __name__ == '__main__':
    opts = parse_args()
    vis(
        opts.dataset,
        opts.detections,
        opts.thresh,
        opts.output_dir,
        sampleNum=opts.sampleNum,
        class_list_file=opts.class_list_file
    )
