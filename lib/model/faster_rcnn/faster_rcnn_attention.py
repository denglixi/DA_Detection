import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
import math
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


class attention_module_multi_head(nn.Module):

    """ Attetion module with vectorized version

    Args:
        roi_feat: [num_rois, feat_dim]
        position_embedding: [num_rois, nongt_dim, emb_dim]
        nongt_dim:
        fc_dim: should be same as group
        feat_dim: dimension of roi_feat, should be same as dim[2]
        dim: a 3-tuple of (query, key, output)
        group:
        index:

    Returns:
        output: [num_rois, ovr_feat_dim, output_dim]
    """

    def __init__(self,
                 nongt_dim, fc_dim, feat_dim,
                 dim=(1024, 1024, 1024),
                 group=16, index=1):
        super(attention_module_multi_head, self).__init__()

        self.nongt_dim = nongt_dim
        self.fc_dim = fc_dim
        self.feat_dim = feat_dim
        self.dim = dim
        self.group = group
        self.index = index

        self.position_fc = nn.Linear(64, self.fc_dim)
        self.querry_fc = nn.Linear(1024, self.dim[0])
        self.k_fc = nn.Linear(1024, self.dim[1])

        self.conv = nn.Conv2d(
            in_channels=16384, out_channels=self.dim[2],
            kernel_size=(1, 1), groups=self.fc_dim)

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

    def forward(self, roi_feat, position_embedding):
        dim_group = (self.dim[0] / self.group, self.dim[1] /
                     self.group, self.dim[2] / self.group)
        nongt_roi_feat = torch.narrow(
            roi_feat, dim=0, start=0, length=self.nongt_dim)
        # [num_rois * nongt_dim, emb_dim]
        position_embedding_reshape = torch.reshape(
            position_embedding, shape=(-1, position_embedding.shape[2]))
        # position_feat_1, [num_rois * nongt_dim, fc_dim]
        position_feat_1 = self.position_fc(position_embedding_reshape)
        position_feat_1_relu = F.relu(position_feat_1)
        # aff_weight, [num_rois, nongt_dim, fc_dim]
        aff_weight = torch.reshape(
            position_feat_1_relu, shape=(-1, self.nongt_dim, self.fc_dim))
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = torch.transpose(aff_weight, 2, 1)

        # multi head
        assert self.dim[0] == self.dim[1], 'Matrix multiply requires same dimensions!'
        q_data = self.querry_fc(roi_feat)
        q_data_batch = torch.reshape(
            q_data, shape=(-1, self.group, dim_group[0]))
        q_data_batch = torch.transpose(q_data_batch, 1, 0)

        k_data = self.k_fc(nongt_roi_feat)
        k_data_batch = torch.reshape(
            k_data, shape=(-1, self.group, dim_group[1]))
        k_data_batch = torch.transpose(k_data_batch, 1, 0)
        v_data = nongt_roi_feat
        # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
        aff = torch.bmm(
            q_data_batch, torch.transpose(k_data_batch, 1, 2))
        # aff_scale, [group, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        aff_scale = torch.transpose(aff_scale, 0, 1)

        assert self.fc_dim == self.group, 'fc_dim != group'
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
        weighted_aff = torch.log(torch.max(
            aff_weight, torch.cuda.FloatTensor(aff_weight.shape).fill_(0) + 1e-6)) + aff_scale
        aff_softmax = F.softmax(
            weighted_aff, dim=2)
        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = torch.reshape(
            aff_softmax, shape=(-1, self.nongt_dim))
        # output_t, [num_rois * fc_dim, feat_dim]
        output_t = torch.mm(aff_softmax_reshape, v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = torch.reshape(
            output_t, shape=(-1, self.fc_dim * self.feat_dim, 1, 1))
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.conv(output_t)
        output = torch.squeeze(linear_out)
        return output


class _fasterRCNNAttention(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, lc, gc):
        super(_fasterRCNNAttention, self).__init__()
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

        self.RFCN_psroi_pool = None

        self.grid_size = cfg.POOLING_SIZE * \
            2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # attention
        # self.conv_new_1 = nn.Conv2d(2048, 256, 1)
        # self.fc_new_1 = nn.dense(name='fc1_new_1', num_hidden=1024)
        # self.fc_new_2 = nn.dense(name='fc2_new_2', num_hidden=1024)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.nongt_dim = 300 if self.training else cfg.TEST.RPN_POST_NMS_TOP_N
        self.attention_1 = attention_module_multi_head(nongt_dim=self.nongt_dim,
                                                       fc_dim=16, feat_dim=1024, index=1, group=16,
                                                       dim=(1024, 1024, 1024))

        self.attention_2 = attention_module_multi_head(nongt_dim=self.nongt_dim,
                                                       fc_dim=16, feat_dim=1024, index=2, group=16,
                                                       dim=(1024, 1024, 1024))

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
        # position_mat, [num_rois, nongt_dim, 4]
        feat_range = torch.arange(0, feat_dim / 8).cuda()
        dim_mat = torch.pow(torch.cuda.FloatTensor(1,).fill_(wave_length),
                            (8. / feat_dim) * feat_range)

        dim_mat = torch.reshape(dim_mat, shape=(1, 1, 1, -1))
        position_mat = torch.unsqueeze(100.0*position_mat, dim=3)
        div_mat = position_mat / dim_mat

        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)

        # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
        embedding = torch.cat((sin_mat, cos_mat), dim=3)
        # embedding, [num_rois, nongt_dim, feat_dim]
        embedding = torch.reshape(embedding, shape=(
            embedding.shape[0], embedding.shape[1], feat_dim))
        return embedding

    @staticmethod
    def extract_position_matrix(bbox, nongt_dim):
        """ Extract position matrix

        Args:
            bbox: [num_boxes, 4]

        Returns:
            position_matrix: [num_boxes, nongt_dim, 4]
        """
        xmin, ymin, xmax, ymax = torch.chunk(bbox,
                                             chunks=4,
                                             dim=1)
        # [num_fg_classes, num_boxes, 1]
        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # [num_fg_classes, num_boxes, num_boxes]
        # delta_x = mx.sym.broadcast_minus(lhs=center_x,
        #                                 rhs=mx.sym.transpose(center_x))
        delta_x = center_x - torch.transpose(center_x, 0, 1)
        delta_x = delta_x / bbox_width
        delta_x = torch.log(torch.max(torch.abs(delta_x), 1e-3 +
                                      torch.cuda.FloatTensor(delta_x.shape).fill_(0)))

        delta_y = center_y - torch.transpose(center_y, 0, 1)
        delta_y = delta_y / bbox_height
        delta_y = torch.log(torch.max(torch.abs(delta_y), 1e-3 +
                                      torch.cuda.FloatTensor(delta_x.shape).fill_(0)))
        delta_width = bbox_width / torch.transpose(bbox_width, 0, 1)
        delta_width = torch.log(delta_width)

        delta_height = bbox_height / torch.transpose(bbox_height, 0, 1)
        delta_height = torch.log(delta_width)

        concat_list = [delta_x, delta_y, delta_width, delta_height]
        for idx, sym in enumerate(concat_list):
            sym = torch.narrow(sym, dim=1, start=0, length=nongt_dim)
            concat_list[idx] = torch.unsqueeze(sym, dim=2)
        position_matrix = torch.cat(concat_list, dim=2)
        return position_matrix

    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=False, eta=1.0):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        if self.lc:
            d_pixel, _ = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))
            # print(d_pixel)
            if not target:
                _, feat_pixel = self.netD_pixel(base_feat1.detach())
        else:
            d_pixel = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))
        base_feat = self.RCNN_base2(base_feat1)
        if self.gc:
            domain_p, _ = self.netD(grad_reverse(base_feat, lambd=eta))
            if target:
                return d_pixel, domain_p  # , diff
            _, feat = self.netD(base_feat.detach())
        else:
            domain_p = self.netD(grad_reverse(base_feat, lambd=eta))
            if target:
                return d_pixel, domain_p  # ,diff
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes)


        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            # rois_label is sub classlable

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

        # return roi_data
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

        elif cfg.POOLING_MODE == 'pspool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # (256,2048,7,7)


        # nongt_dim = cfg.TRAIN.RPN_POST_NMS_TOP_N if self.training else cfg.TEST.RPN_POST_NMS_TOP_N
        nongt_dim = 300 if self.training else cfg.TEST.RPN_POST_NMS_TOP_N

        position_matrix = self.extract_position_matrix(
            rois.view(-1, 5)[:, :4].clone(), nongt_dim=nongt_dim)
        position_embedding = self.extract_position_embedding(
            position_matrix, feat_dim=64)

        pooled_feat = self.fc1(pooled_feat)
        attention_feat_1 = self.attention_1(pooled_feat, position_embedding)
        pooled_feat = pooled_feat + attention_feat_1
        pooled_feat = self.fc2(pooled_feat)
        attention_feat_2 = self.attention_2(pooled_feat, position_embedding)
        pooled_feat = pooled_feat + attention_feat_2

        #feat_pixel = torch.zeros(feat_pixel.size()).cuda()
        if self.lc:
            feat_pixel = feat_pixel.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat_pixel, pooled_feat), 1)
        if self.gc:
            feat = feat.view(1, -1).repeat(pooled_feat.size(0), 1)
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(
                rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

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

        if self.training:
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, d_pixel, domain_p  # ,diff

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

        normal_init(self.fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.attention_1.querry_fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.attention_1.position_fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.attention_1.k_fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.attention_1.conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.attention_2.querry_fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.attention_2.position_fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.attention_2.k_fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.attention_2.conv, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
