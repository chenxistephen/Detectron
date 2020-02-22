# Written by Stephen Xi Chen for evaluating COCO json datasets
# Last updated: 11/17/2019

##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from collections import OrderedDict
import cv2
import datetime
import logging
import numpy as np
import os

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.core.rpn_generator import generate_rpn_on_dataset
from detectron.core.rpn_generator import generate_rpn_on_range
from detectron.core.test import im_detect_all
from detectron.datasets import task_evaluation
from detectron.datasets.json_dataset import JsonDataset
from detectron.modeling import model_builder
from detectron.utils.io import save_object, load_object
from detectron.utils.timer import Timer
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu
import detectron.utils.net as net_utils
import detectron.utils.subprocess as subprocess_utils
import detectron.utils.vis as vis_utils

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating a dataset')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--eval_test', dest='eval_test', help='eval_test', action='store_true'
    )
    parser.add_argument(
        '--compute_loc_pr', dest='compute_loc_pr', help='compute_loc_pr', action='store_true'
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def evaluation_on_dataset(
    dataset_name,
    output_dir,
    test_only=True,
    computeLocPR=False
):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    ################################################################
    import pickle
    res_file = os.path.join(
        output_dir, 'bbox_' + dataset_name + '_results.json'
    )
    print ("res_file = {}==========================".format(res_file))
    if os.path.exists(res_file):
        import detectron.datasets.json_dataset_evaluator as json_dataset_evaluator
        print ("res_file = {} exists! Loading res_file".format(res_file))
        coco_eval = json_dataset_evaluator._do_detection_eval(dataset, res_file, output_dir,computeLocPR)
        box_results = task_evaluation._coco_eval_to_box_results(coco_eval)
        results = OrderedDict([(dataset.name, box_results)])
        return results     
    ################################################################
    det_name = "detections.pkl"
    det_file = os.path.join(output_dir, det_name)
    print ("det_file = {}==========================".format(det_file))
    if os.path.exists(det_file):
        print ("{} exists! Loading detection results".format(det_file))
        res = load_object(det_file) #pickle.load(open(det_file, 'rb'))
        all_boxes = res['all_boxes']
        all_segms = res['all_segms']
        all_keyps = res['all_keyps']
    ################################################################
    # elif multi_gpu:
    #     num_images = len(dataset.get_roidb())
    #     all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
    #         weights_file, dataset_name, proposal_file, num_images, output_dir
    #     )
    # else:
    #     all_boxes, all_segms, all_keyps = test_net(
    #         weights_file, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
    #     )
    # test_timer.toc()
    # logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    # if test_only:
    #     return OrderedDict([(dataset.name, all_boxes)])
    # else:
    results = task_evaluation.evaluate_all(
        dataset, all_boxes, all_segms, all_keyps, output_dir,computeLocPR=computeLocPR
    )
    return results