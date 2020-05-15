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
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse
from model.faster_rcnn.global_local_domain_classifier import conv3x3, conv1x1, netD_pixel, netD_dc, netD
from model.faster_rcnn.faster_rcnn_global_local_backbone import FasterRCNN


class image_weakly_moudle(nn.Module):
    def __init__(self, input_dim, output_dim=20):
        super(image_weakly_moudle, self).__init__()
        self.conv1 = conv3x3(input_dim, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


class FasterRCNN_Teacher_Student(FasterRCNN):
    def __init__(self, classes, class_agnostic, lc, gc, backbone_type='res101',
                 pretrained=False, weakly_type='max',
                 is_uda=False, is_img_wda=True, is_region_wda=True):
        super(FasterRCNN_Teacher_Student, self).__init__(
            classes, class_agnostic, lc, gc, backbone_type, pretrained)
        self.weakly_type = weakly_type
        self.image_weakly = image_weakly_moudle(1024, len(self.classes))
        self.is_uda = is_uda
        self.is_img_wda = is_img_wda
        self.is_region_wda = is_region_wda
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(600)
        ])
    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=False, teacher=False, eta=1.0):
        DEBUG = False
        if DEBUG:
            print("debug........")

        batch_size = im_data.size(0)

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
            num_target_img = 2
            im_data = im_data.view(num_target_img, 3, 512,512)
            im_info = im_info.repeat(num_target_img, 1 )
            gt_boxes = gt_boxes.repeat(num_target_img, 1,1)
            num_boxes = num_boxes.repeat(num_target_img)


        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        if self.is_uda:
            d_local, d_local_context = self.foward_local_domain_cls(
                base_feat1, eta, target)
        else:
            d_local = None
            d_local_context = None

        base_feat = self.RCNN_base2(base_feat1)
        if self.is_uda:
            d_global, d_global_context = self.foward_global_domain_cls(
            base_feat, eta, target)
        else:
            d_global = None
            d_global_context = None


        if self.training and target or DEBUG:
            if self.is_img_wda:
                image_multi_cls_score = self.image_weakly(base_feat).squeeze()
                img_BCE_loss = F.binary_cross_entropy(
                    F.sigmoid(image_multi_cls_score), cls_label)
            else:
                img_BCE_loss = None

        # feed base feature map tp RPN to obtain rois
        # ignore gt_box of  target
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes, target=target, force_test_mode=target)
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
        if self.is_uda:
            pooled_feat = self.forward_add_context(
                d_local_context, d_global_context, pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        bbox_pred = self.bbox_pred_norm(
            bbox_pred, rois_label, force_test_mode=target)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        cls_num = cls_prob.shape[-1]

        # for test only

        if DEBUG:
            DEBUG_weakly_type = 'max'
            if DEBUG_weakly_type == 'max':
                cls_prob = cls_prob.view(2, -1, cls_num)
                max_roi_cls_prob = torch.max(cls_prob[0], 0)[0]
                max_zero = torch.zeros_like(max_roi_cls_prob)
                max_one = torch.ones_like(max_roi_cls_prob)
                select_max = torch.where(max_roi_cls_prob > 0.2, max_one, max_zero)
            else:
                selected_rois_index =  torch.argmax(cls_score, 0)
                selected_rois_index = torch.unique(selected_rois_index.cpu()).cuda()
                selected_rois_score = cls_score[selected_rois_index]
                sum_selected_rois_score = torch.sum(selected_rois_score, 0)
                sum_selected_rois_prob = F.sigmoid(sum_selected_rois_score)
                max_zero = torch.zeros_like(sum_selected_rois_score)
                max_one = torch.ones_like(sum_selected_rois_score)
                select_max = torch.where(max_roi_cls_prob > 0.2, max_one, max_zero)
            print(select_max)
            print(cls_label)
            BCE_loss = F.binary_cross_entropy(
                max_roi_cls_prob, cls_label)
            print(BCE_loss)
            pdb.set_trace()

        # weakly region-level domain adapatation
        if self.training and target or DEBUG:
            #Consistency loss

            if self.is_region_wda:
                import pdb
                pdb.set_trace()
                if self.weakly_type == 'max':
                    cls_prob = cls_prob.view(2, -1, cls_num)
                    cls_prob_teacher = cls_prob[0]
                    cls_prob_student = cls_prob[1]
                    max_roi_cls_prob_student = torch.max(cls_prob_student, 0)[0]
                    max_roi_cls_prob_teacher = torch.max(cls_prob_teacher, 0)[0]
                    #max_roi_cls_prob = torch.max(cls_prob, 0)[0]
                    BCE_loss = F.binary_cross_entropy(
                        max_roi_cls_prob, cls_label)
                    Weakly_Consistency_L2_loss = F.mse(cls_prob_student, cls_prob_teacher)
                elif self.weakly_type == 'select_max':
                    #max_roi_cls_prob = torch.max(cls_prob, 0)[0]
                    selected_rois_index =  torch.argmax(cls_score, 0)
                    selected_rois_index = torch.unique(selected_rois_index.cpu()).cuda()
                    selected_rois_score = cls_score[selected_rois_index]
                    sum_selected_rois_score = torch.sum(selected_rois_score, 0)
                    #sum_selected_rois_score = torch.clamp(sum_selected_rois_score, 0., 1.)
                    #sum_selected_rois_score = torch.clamp(sum_selected_rois_score, 0.00000001, 0.9999999999)
                    #if not ((sum_selected_rois_score.data.cpu().numpy() >= 0.).all() and (sum_selected_rois_score.data.cpu().numpy() <= 1.).all()):
                    #    pdb.set_trace()
                    #if not ((cls_label.data.cpu().numpy() >= 0.).all() and (cls_label.data.cpu().numpy() <= 1.).all()):
                    #    pdb.set_trace()
                    BCE_loss = F.binary_cross_entropy_with_logits(
                        sum_selected_rois_score, cls_label)
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
                elif self.weakly_type == 'rank':
                    BCE_loss = None
            else:
                BCE_loss = None
            return d_local, d_global, img_BCE_loss, BCE_loss, Weakly_Consistency_L2_loss
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
