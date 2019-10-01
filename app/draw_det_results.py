#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 denglixi <denglixi@xgpd0>
#
# Distributed under terms of the MIT license.

"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.prefood_res50_attention import PreResNet50Attention
from datasets.food_category import get_categories
from datasets.id2name import id2chn, id2eng
from datasets.sub2main import sub2main_dict
from datasets.voc_eval import get_gt_recs
from model.utils.parser_func import parse_args, set_dataset_args
from datasets.food_category import get_categories
from functools import partial

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def get_det_bbox_with_cls_of_img(all_boxes, img_idx):
    """get_det_bbox_with_cls_of_img

    :param all_boxes: det result
    :param img_idx: img idx in imageset.txt
    :return bboxes: N*6-dim matrix, N is number of bbox, 6 is [x1,y1,x2,y2,score,cls], the cls is the addtion information we get from this function
    """
    # get all box of img
    img_all_boxes = [b[img_idx] for b in all_boxes]
    # add cls_idx to box_cls
    bboxes = None
    for cls_idx_img, boxes_of_cls in enumerate(img_all_boxes):
        if len(boxes_of_cls) != 0:
            cls_cloumn = np.zeros(len(boxes_of_cls)) + cls_idx_img
            # img_all_boxes[cls_idx] = np.c_[boxes_of_cls, cls_cloumn]
            if bboxes is None:
                bboxes = np.c_[boxes_of_cls, cls_cloumn]
            else:
                bboxes = np.vstack(
                    (bboxes, np.c_[boxes_of_cls, cls_cloumn]))
    return bboxes


def get_main_cls(sub_classes):
    main_classes = []
    for i in sub_classes:
        main_classes.append(sub2main_dict[i])
    main_classes = set(main_classes)
    main_classes = list(main_classes)
    return main_classes


def pred_boxes_regression(boxes, bbox_pred, scores, classes, cfg, args):
    """pred_boxes_regression

    :param boxes: rois from RPN
    :param bbox_pred: regression result of each bbox
    :param scores: score of bboxes
    :param classes: classes of result
    :param cfg: cfg
    :param args: args
    """
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(
                    1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = torch.from_numpy(
            np.tile(boxes, (1, scores.shape[2]))).cuda()

    pred_boxes /= data[1][0][2].item()
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    return scores, pred_boxes


def get_all_boxes(det_file_path):
    """get_all_boxes

    :param det_file_path: path of the 'detection.pkl' file

    """
    with open(det_file_path, 'rb') as f:
        all_boxes = pickle.load(f)
    return all_boxes


def show_image(im, bboxes, gt_cls, imdb):
    count2color = {
        1: (219, 224, 5),
        2: (64, 192, 245),
        3: (40, 206, 165),
        4: (120, 208, 91),
        5: (211, 132, 69),
        6: (253, 182, 49)
    }

    false_color = (0, 0, 233)

    im2show = np.copy(im)
    color_count = 0
    for b_i in range(len(bboxes)-1, -1, -1):
        if bboxes[b_i, 5] in gt_cls:
            color = count2color[color_count % len(count2color) + 1]
            color_count += 1
        else:
            color = false_color
        im2show = vis_detections(
            im2show,
            id2eng[
                imdb.classes[
                    int(bboxes[b_i, 5])
                ]
            ],
            np.array([bboxes[b_i, :5], ]),
            color=color,
            is_show_text=False
        )

    return im2show


def cal_top5_accury(bbox, gt_clses_ind):
    if bbox is None or len(bbox) == 0 or len(gt_clses_ind) == 0:
        return 0, 0
    det_clses_ind = bbox[:, 5]
    precision = len(set(det_clses_ind) & set(
        gt_clses_ind)) / len(set(det_clses_ind))
    recall = len(set(det_clses_ind) & set(gt_clses_ind)) / \
        len(set(gt_clses_ind))

    return recall, precision


def get_boxes_cls_ind(i, all_boxes, threshold=0.5):
    bboxes = get_det_bbox_with_cls_of_img(all_boxes, i)
    if bboxes is not None:
        bboxes = bboxes[np.where(bboxes[:, 4] >= threshold)]
        _, uni_id = np.unique(bboxes[:, 5], return_index=True)
        bboxes = bboxes[uni_id]
        bboxes = bboxes[bboxes[:, 4].argsort()]
        #bboxes = bboxes[::-1]

    return bboxes


def is_result_ordered(recalls_and_precisions):
    """is_result_ordered

    :param recalls_and_precisions: It's a list. [ ( recall1,, precison1 ) , (recall2, precision2), ... ]
    """
    r_and_p_first = recalls_and_precisions[0]
    for r_and_p in recalls_and_precisions[1:]:
        # need to eliminate the situation that results are equal
        if r_and_p_first[0] > r_and_p[0] or r_and_p_first[1] > r_and_p[1] or (r_and_p_first[0] == r_and_p[0] and r_and_p_first[1] == r_and_p[1]):
            return False
        r_and_p_first = r_and_p
    return True


if __name__ == '__main__':

    args = parse_args()
    args = set_dataset_args(args, test=True)

    test_canteen = args.imdbval_name.split('_')[1]
    #test_canteen = args.dataset.split('_')[1]

    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                     'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))
    num_images = len(imdb.image_index)

    # Get all boxes of each model
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    # get detection results of each model from its pkl file
    ordered_models_results = list(map(get_all_boxes, args.model_result_paths))

    # Get groundtruth
    imagenames, recs = get_gt_recs(
        imdb.cachedir, imdb.imagesetfile, imdb.annopath)

    # recognition file
    if not os.path.exists('case_study'):
        os.makedirs('case_study')
    recognition_f = open("./case_study/recogniton.txt", 'w')

    # comparing results of each image between models, saving the eligible image
    for i in range(num_images):
        # GT
        R = [obj for obj in recs[imagenames[i]]]
        gt_bbox = np.array([x['bbox'] for x in R])
        gt_cls = np.array([x['name'] for x in R])
        try:
            gt_clses_ind = [imdb.classes.index(x) for x in gt_cls]
        except:
            # some class not in imdb.classes, which means that there are some categories are not included in training data, i.e. categories that are less than 10
            continue

        # Get detection with cls information [x1,y1,x2,y2,score, cls_ind]
        get_boxes_cls = partial(get_boxes_cls_ind, i, threshold=0.5)
        orderd_models_boxes_image_i = list(
            map(get_boxes_cls, ordered_models_results))

        # calculate the accuracy and recall of each image in different models
        cal_top5_accury_with_gt = partial(
            cal_top5_accury, gt_clses_ind=gt_clses_ind)
        recalls_precs_models_image_i = list(
            map(cal_top5_accury_with_gt, orderd_models_boxes_image_i))

        # save eligible image
        # construct ordered
        # case1: faster < local < local + global
        # case2: faster < global < local + global

        r_p_case1 = [recalls_precs_models_image_i[0],
                     recalls_precs_models_image_i[1], recalls_precs_models_image_i[3]]
        r_p_case2 = [recalls_precs_models_image_i[0],
                     recalls_precs_models_image_i[2], recalls_precs_models_image_i[3]]

        # if is_result_ordered(recalls_precs_models_image_i):
        if is_result_ordered(r_p_case1) and is_result_ordered(r_p_case2):
            print("image id", i)
            print("image id:{}".format(i), file=recognition_f)
            for r_and_p in recalls_precs_models_image_i:
                print(r_and_p[0], r_and_p[1], file=recognition_f)
            print('-------', file=recognition_f)
            print(gt_clses_ind, file=recognition_f)

            for model_bboxes in orderd_models_boxes_image_i:

                print('------', file=recognition_f)
                # write name
                for b_cls in model_bboxes[:, 5]:
                    recognition_f.write(
                        id2eng[imdb.classes[int(b_cls)]] + '\t')
                recognition_f.write('\n')

                # write score
                for b_cls in model_bboxes[:, 4]:
                    recognition_f.write("{:.2f}".format(b_cls) + '\t')
                recognition_f.write('\n')

                for b_cls in model_bboxes[:, 5]:
                    if b_cls in gt_clses_ind:
                        recognition_f.write('Ture\t')
                    else:
                        recognition_f.write('False\t')
                recognition_f.write('\n')

            # def show_image(im, bboxes, gt_cls, imdb, color=(233,174,61)):
            # BGR

            if True:
                im = cv2.imread(imdb.image_path_at(i))
                count = 0
                # , transfer_boxes, relation_boxes]:
                for model_bboxes_imagei in orderd_models_boxes_image_i:
                    count += 1
                    im2show = show_image(
                        im, model_bboxes_imagei, gt_clses_ind, imdb)
                    # exit()
                    cv2.imwrite(
                        "./case_study/{}_{}.jpg".format(i, count), im2show)
                    #cv2.namedWindow("frame", 0)
                    #cv2.resizeWindow("frame", 1700, 900)
                    #cv2.imshow('frame', im2show)
                    # cv2.waitKey(0)
