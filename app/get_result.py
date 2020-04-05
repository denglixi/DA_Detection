#! /usr/bin/env python
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
from datasets.food_category import get_categories
from datasets.id2name import id2chn, id2eng
from datasets.sub2main import sub2main_dict
from datasets.voc_eval import get_gt_recs
from test_net import get_data2imdbval_dict

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


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--imgset', dest='imgset',
                        help='val, test',
                        default='val', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--test_cache', dest='test_cache',
                        action='store_true')
    parser.add_argument('--save_for_vis', dest='save_for_vis',
                        type=bool, default=False)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def set_imdbval_name(args):
    data2imdbval_dict = get_data2imdbval_dict(args.imgset)
    args.imdbval_name = data2imdbval_dict[args.dataset]
    return args


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


def get_all_boxes(save_name, imdb):
    """get_all_boxes

    :param save_name:  faster_rcnn_{}_{}_{}
                        args.checksession, args.checkepoch, args.checkpoint)
    :param imdb:get outputdir
    """
    output_dir = get_output_dir(imdb, save_name)
    det_file = os.path.join(
        output_dir, 'detections.pkl')
    with open(det_file, 'rb') as f:
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
        )

    return im2show


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)
    args = set_imdbval_name(args)

    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    # args = construct_dataset(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False

    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        args.imdbval_name, False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))
    num_images = len(imdb.image_index)

    # Get all boxes of each model
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    baseline_save_name = 'faster_rcnn_{}_{}_{}'.format(
        args.checksession, args.checkepoch, args.checkpoint)
    transfer_save_name = 'faster_rcnn_{}_{}_{}'.format(
        444, 32, args.checkpoint)
    relation_save_name = 'faster_rcnn_{}_{}_{}'.format(
        5, 14, args.checkpoint)

    baseline_all_boxes = get_all_boxes(baseline_save_name, imdb)
    transfer_all_boxes = get_all_boxes(transfer_save_name, imdb)
    relation_all_boxes = get_all_boxes(relation_save_name, imdb)

    # Get groundtruth
    imagenames, recs = get_gt_recs(
        imdb.cachedir, imdb.imagesetfile, imdb.annopath)

    def cal_top5_accury(bbox, gt_clses_ind):
        if bbox is None or len(bbox) == 0 or len(gt_clses_ind) == 0:
            return 0, 0
        det_clses_ind = bbox[:, 5]
        precision = len(set(det_clses_ind) & set(
            gt_clses_ind)) / len(set(det_clses_ind))
        recall = len(set(det_clses_ind) & set(gt_clses_ind)) / \
            len(set(gt_clses_ind))

        return recall, precision

    with open('./co_cls.txt', 'r') as f:
        co_matrix_cls = [x.strip() for x in f.readlines()]

    def get_boxes_cls_ind(i, all_boxes, threshold=0.5):
        bboxes = get_det_bbox_with_cls_of_img(all_boxes, i)
        if bboxes is not None:
            bboxes = bboxes[np.where(bboxes[:, 4] >= threshold)]
            _, uni_id = np.unique(bboxes[:, 5], return_index=True)
            bboxes = bboxes[uni_id]
            bboxes = bboxes[bboxes[:, 4].argsort()]
            #bboxes = bboxes[::-1]

        return bboxes

    recognition_f = open("./case_study/recogniton.txt", 'w')
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

        # Get det cls_index
        baseline_boxes = get_boxes_cls_ind(i, baseline_all_boxes)
        transfer_boxes = get_boxes_cls_ind(i, transfer_all_boxes)
        relation_boxes = get_boxes_cls_ind(i, relation_all_boxes)

        if i == 505:
            gt_clses_ind.append(1)
        elif i == 330:
            gt_clses_ind.append(43)

        b_recall, b_prec = cal_top5_accury(baseline_boxes, gt_clses_ind)
        t_recall, t_prec = cal_top5_accury(transfer_boxes, gt_clses_ind)
        r_rec, r_pre = cal_top5_accury(relation_boxes, gt_clses_ind)

        # if (t_recall > b_recall and t_prec >= b_prec and r_rec > t_recall and r_pre > t_prec:

        if ((t_recall > b_recall and t_prec >= b_prec) or (t_recall >= b_recall and t_prec > b_prec)) and ((r_rec > t_recall and r_pre >= t_prec) or (r_rec >= t_recall and r_pre > t_prec)):
            print("image id", i)
            print("image id:{}".format(i), file=recognition_f)
            print(b_recall, b_prec, file=recognition_f)
            print(t_recall, t_prec, file=recognition_f)
            print(r_rec, r_pre, file=recognition_f)
            print('-------', file=recognition_f)
            print(gt_clses_ind, file=recognition_f)

            a = list(zip(baseline_boxes[:, 5], baseline_boxes[:, 4]))

            for bboxes in (baseline_boxes, transfer_boxes, relation_boxes):
                print('------', file=recognition_f)

                # write name
                for b_cls in bboxes[:, 5]:
                    recognition_f.write(
                        id2eng[imdb.classes[int(b_cls)]] + '\t')
                recognition_f.write('\n')

                # write score
                for b_cls in bboxes[:, 4]:
                    recognition_f.write("{:.2f}".format(b_cls) + '\t')
                recognition_f.write('\n')

                for b_cls in bboxes[:, 5]:
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
                for bboxes_i in [baseline_boxes, transfer_boxes, relation_boxes]:
                    count += 1
                    im2show = show_image(im, bboxes_i, gt_clses_ind, imdb)
                    # exit()
                    cv2.imwrite(
                        "./case_study/{}_{}.jpg".format(i, count), im2show)
                    # cv2.namedWindow("frame", 0)
                    # cv2.resizeWindow("frame", 1700, 900)
                    # cv2.imshow('frame', im2show)
                    # cv2.waitKey(0)
