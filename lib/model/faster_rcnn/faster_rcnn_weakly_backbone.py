#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 lxdeng <lxdeng@next-gpu1>
#
# Distributed under terms of the MIT license.

"""

"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse

from model.faster_rcnn.global_local_domain_classifier import conv3x3, conv1x1, netD_pixel, netD_dc, netD
from model.faster_rcnn.faster_rcnn_global_local_backbone import FasterRCNN


class FasterRCNN_Weakly(FasterRCNN):
    def __init__(self, classes, class_agnostic, lc, gc, backbone_type='res101', pretrained=False, weakly_type='max'):
        super(FasterRCNN_Weakly, self).__init__(
            classes, class_agnostic, lc, gc, backbone_type, pretrained)
        self.weakly_type = weakly_type

    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=False,  eta=1.0):
        DEBUG = False
        if DEBUG:
            print("debug........")

        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # get all vector of class for label
        if self.training and target or DEBUG:
            cls_label_ind = torch.unique(gt_boxes[:, :, 4].cpu())
            cls_label_ind = cls_label_ind[cls_label_ind <
                                          self.n_classes]
            cls_label = torch.zeros(self.n_classes)
            cls_label[cls_label_ind.long()] = 1
            # assume always have backgound categories
            cls_label[0] = 1
            cls_label = cls_label.cuda()
            cls_label.clamp(0, 1)
            cls_label.requires_grad = False

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        d_local, d_local_context = self.foward_local_domain_cls(
            base_feat1, eta, target)

        base_feat = self.RCNN_base2(base_feat1)
        d_global, d_global_context = self.foward_global_domain_cls(
            base_feat, eta, target)

        # feed base feature map tp RPN to obtain rois
        # ignore gt_box of  target
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes, force_test_mode=target)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = self.forward_region_proposal(
            rois, gt_boxes, num_boxes, force_test_mode=target)
        if not self.training:
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # do roi pooling based on predicted rois
        pooled_feat = self.forward_roi_pooling(rois, base_feat)

        # feed pooled features to top model
        # pooled_feat = self._head_to_tail(pooled_feat)
        pooled_feat = self.RCNN_top(pooled_feat).mean(3).mean(2)

        # add context based on gc and lc hyperparameter
        pooled_feat = self.forward_add_context(
            d_local_context, d_global_context, pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        bbox_pred = self.bbox_pred_norm(
            bbox_pred, rois_label, force_test_mode=target)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        # for test only

        if DEBUG:
            max_roi_cls_prob = torch.max(cls_prob, 0)[0]
            max_zero = torch.zeros_like(max_roi_cls_prob)
            max_one = torch.ones_like(max_roi_cls_prob)

            select_max = torch.where(max_roi_cls_prob > 0.2, max_one, max_zero)
            print(select_max)
            print(cls_label)
            BCE_loss = F.binary_cross_entropy(
                max_roi_cls_prob, cls_label)
            print(BCE_loss)

            pdb.set_trace()
        if self.training and target:
            if self.weakly_type == 'max':
                #     c1 c2 c3
                # b1   1
                # b2
                # b3
                #cls_prob_sum = torch.sum(cls_prob, 0)
                # x = max(1, x)
                #cls_prob_sum = cls_prob_sum.repeat(2, 1)
                #cls_prob_sum = torch.min(cls_prob_sum, 0)[0]
                max_roi_cls_prob = torch.max(cls_prob, 0)[0]
                #assert (max_roi_cls_prob.data.cpu().numpy().all() >= 0. and max_roi_cls_prob.data.cpu().numpy().all() <= 1.)
                if not (max_roi_cls_prob.data.cpu().numpy().all() >= 0. and max_roi_cls_prob.data.cpu().numpy().all() <= 1.):
                    pdb.set_trace()
                if not (cls_label.data.cpu().numpy().all() >= 0. and cls_label.data.cpu().numpy().all() <= 1.):
                    pdb.set_trace()

                max_zero = torch.zeros_like(max_roi_cls_prob)
                max_one = torch.ones_like(max_roi_cls_prob)
                select_max = torch.where(
                    max_roi_cls_prob > 0.2, max_one, max_zero)
                BCE_loss = F.binary_cross_entropy(
                    max_roi_cls_prob, cls_label)
                return d_local, d_global, BCE_loss
            elif self.weakly_type == 'sum':
                cls_score_t = cls_score.transpose(0, 1)
                weight_of_roi_in_each_cls = F.softmax(cls_score_t, 1)
                weight_of_roi_in_each_cls = weight_of_roi_in_each_cls.transpose(
                    0, 1)
                weighted_prob = torch.mul(cls_prob, weight_of_roi_in_each_cls)
                weighted_prob_sum = weighted_prob.sum(0)
                # To eliminate the error on bce loss at begining while some value >= 1
                weighted_prob_sum = torch.clamp(weighted_prob_sum, 0, 1)
                #print(weighted_prob_sum.min(), weighted_prob_sum.max())
                pdb.set_trace()
                #print(cls_label.min(), cls_label.max())
                print(weighted_prob_sum, cls_label)
                BCE_loss = F.binary_cross_entropy(
                    weighted_prob_sum, cls_label)
                return d_local, d_global, BCE_loss
            elif self.weakly_type == 'rank':
                BCE_loss = None
                return d_local, d_global, BCE_loss

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, d_local, d_global  # ,diff
