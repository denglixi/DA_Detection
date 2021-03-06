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


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, lc, gc):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(
            cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(
            cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * \
            2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=False, eta=1.0):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # get all vector of class for label
        if self.training and target:
            cls_label_ind = torch.unique(gt_boxes[:, :, 4].cpu())
            cls_label = torch.zeros(self.n_classes)
            cls_label[cls_label_ind.long()] = 1
            # assume always have backgound categories
            cls_label[0] = 1
            cls_label = cls_label.cuda()
            cls_label.requires_grad = False

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        if self.lc:
            d_pixel, _ = self.netD_pixel_1(grad_reverse(base_feat1, lambd=eta))
            # print(d_pixel)
            if not target:
                _, feat_pixel = self.netD_pixel_1(base_feat1.detach())
        else:
            d_pixel = self.netD_pixel_1(grad_reverse(base_feat1, lambd=eta))

        base_feat2 = self.RCNN_base2(base_feat1)
        if self.lc:
            d_pixel_2, _ = self.netD_pixel_2(
                grad_reverse(base_feat2, lambd=eta))
        else:
            d_pixel_2 = self.netD_pixel_2(grad_reverse(base_feat2, lambd=eta))

        base_feat3 = self.RCNN_base3(base_feat2)
        if self.lc:
            d_pixel_3, _ = self.netD_pixel_3(
                grad_reverse(base_feat3, lambd=eta))
        else:
            d_pixel_3 = self.netD_pixel_3(grad_reverse(base_feat3, lambd=eta))
            # print(d_pixel_3.mean())

        base_feat4 = self.RCNN_base4(base_feat3)
        if self.gc:
            d_pixel_4, _ = self.netD_1(grad_reverse(base_feat4, lambd=eta))
        else:
            d_pixel_4 = self.netD_1(grad_reverse(base_feat4, lambd=eta))

        # something wrong
        base_feat = self.RCNN_base5(base_feat4)
        # for target domain training, we need to return the d_pixel, domain_p
        if self.gc:
            domain_p, _ = self.netD(grad_reverse(base_feat, lambd=eta))
            if target:
                return d_pixel, d_pixel_2, d_pixel_3, d_pixel_4, domain_p
            _, feat = self.netD(base_feat.detach())
        else:
            domain_p = self.netD(grad_reverse(base_feat, lambd=eta))
            if target:
                return d_pixel, d_pixel_2, d_pixel_3, d_pixel_4, domain_p

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training and not target:
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
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(
                rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(
                base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        #feat_pixel = torch.zeros(feat_pixel.size()).cuda()
        if self.lc:
            feat_pixel = feat_pixel.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat_pixel, pooled_feat), 1)
        if self.gc:
            feat = feat.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat, pooled_feat), 1)
            # compute bbox offset

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic and not target:
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(
                rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        # compute the sum of weakly score
        if False:
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
            BCE_loss = F.binary_cross_entropy(max_roi_cls_prob, cls_label)
            return d_pixel, domain_p, BCE_loss

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        # for weakly detection, concentrate the cls_score and calculate the loss

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        # return d_pixel, d_pixel_2, d_pixel_3, d_pixel_4, domain_p
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, d_pixel, d_pixel_2, d_pixel_3, d_pixel_4, domain_p  # ,diff

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

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
