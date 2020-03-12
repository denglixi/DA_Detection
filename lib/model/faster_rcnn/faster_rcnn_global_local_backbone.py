import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse

from model.faster_rcnn.global_local_domain_classifier import conv3x3, conv1x1, netD_pixel, netD_dc, netD
from model.faster_rcnn.resnet_global_local_backbone import resnet_backbone


class FasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, lc, gc, backbone_type='res101', pretrained=False):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        self.backbone_type = backbone_type
        self.pretrained = pretrained

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc

    def _init_modules(self):

        # define backbone
        if self.backbone_type == 'vgg':
            pass
        elif self.backbone_type == 'res101':
            backbone_net = resnet_backbone(
                num_layers=101, pretrained=self.pretrained)
        self.RCNN_base1, self.RCNN_base2, self.RCNN_top = backbone_net.init_modules()
        self.dout_base_model = backbone_net.dout_base_model

        # cross domain classifier
        self.netD_pixel = netD_pixel(
            input_dim=256, output_dim=128, context=self.lc)
        self.netD = netD(input_dim=1024, context=self.gc)

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        # define RCNN pred
        feat_d = 2048
        if self.lc:
            feat_d += 128
        if self.gc:
            feat_d += 128
        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    def _fix_parameter(self):
        # Fix blocks
        for p in self.RCNN_base1[0].parameters():
            p.requires_grad = False
        for p in self.RCNN_base1[1].parameters():
            p.requires_grad = False

    def foward_global_domain_cls(self, base_feat, eta, target):
        if self.gc:
            domain_global, _ = self.netD(grad_reverse(base_feat, lambd=eta))
            if not target:
                _, feat = self.netD(base_feat.detach())
            else:
                feat = None
        else:
            domain_global = self.netD(grad_reverse(base_feat, lambd=eta))
            feat = None
        return domain_global, feat  # , diff

    def foward_local_domain_cls(self, base_feat, eta, target):

        if self.lc:
            d_pixel, _ = self.netD_pixel(
                grad_reverse(base_feat, lambd=eta))
            # print(d_pixel.mean())
            if not target:
                _, feat_pixel = self.netD_pixel(base_feat.detach())
            else:
                feat_pixel = None
        else:
            d_pixel = self.netD_pixel(grad_reverse(base_feat, lambd=eta))
            feat_pixel = None
        return d_pixel, feat_pixel

    def forward_region_proposal(self, rois, gt_boxes, num_boxes, force_test_mode=False):
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training and not force_test_mode:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(
                rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
        rois = Variable(rois)

        return rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws

    def forward_roi_pooling(self, rois, base_feat):

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        return pooled_feat

    def forward_add_context(self, d_local_context, d_global_context, pooled_feat):
        if self.lc:
            d_local_context = d_local_context.view(
                1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((d_local_context, pooled_feat), 1)
        if self.gc:
            d_global_context = d_global_context.view(
                1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((d_global_context, pooled_feat), 1)
            # compute bbox offset
        return pooled_feat

    def bbox_pred_norm(self, bbox_pred, rois_label, force_test_mode=False):
        if self.training and not self.class_agnostic and not force_test_mode:
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(
                rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)
        return bbox_pred


    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=False, eta=1.0):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        d_local, d_local_context = self.foward_local_domain_cls(
            base_feat1, eta, target)

        base_feat = self.RCNN_base2(base_feat1)
        d_global, d_global_context = self.foward_global_domain_cls(
            base_feat, eta, target)

        if target:
            return d_local, d_global

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = self.forward_region_proposal(
            rois, gt_boxes, num_boxes)
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
        bbox_pred = self.bbox_pred_norm(bbox_pred, rois_label)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

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

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def _set_bn(self):

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.RCNN_base1.apply(set_bn_fix)
        self.RCNN_base2.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base1.eval()
            self.RCNN_base1[4].train()
            self.RCNN_base2.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base1.apply(set_bn_eval)
            self.RCNN_base2.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def create_architecture(self):
        self._init_modules()
        self._set_bn()
        self._fix_parameter()
        self._init_weights()
