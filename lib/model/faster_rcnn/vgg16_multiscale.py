# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_multiscale import _fasterRCNN
# from model.faster_rcnn.faster_rcnn_imgandpixellevel_gradcam  import _fasterRCNN
from model.utils.config import cfg
from model.faster_rcnn.global_local_domain_classifier import netD_pixel, netD

import pdb


class vgg16_multiscale(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, lc=False, gc=False):
        self.model_path = cfg.VGG_PATH
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.lc = lc
        self.gc = gc

        _fasterRCNN.__init__(self, classes, class_agnostic, self.lc, self.gc)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict(
                {k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(
            *list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # print(vgg.features)
        self.RCNN_base1 = nn.Sequential(
            *list(vgg.features._modules.values())[:5])

        self.RCNN_base2 = nn.Sequential(
            *list(vgg.features._modules.values())[5:10])

        self.RCNN_base3 = nn.Sequential(
            *list(vgg.features._modules.values())[10:14])

        self.RCNN_base4 = nn.Sequential(
            *list(vgg.features._modules.values())[14:23])

        self.RCNN_base5 = nn.Sequential(
            *list(vgg.features._modules.values())[23:-1])

        # print(self.RCNN_base1)
        # print(self.RCNN_base2)
        self.netD_pixel_1 = netD_pixel(64,  128, context=self.lc)
        self.netD_pixel_2 = netD_pixel(128, 64, context=self.lc)
        self.netD_pixel_3 = netD_pixel(256, 128, context=self.lc)
        self.netD_pixel_4 = netD_pixel(512, 256, context=self.lc)

        self.netD = netD(context=self.gc)
        self.netD_1 = netD(context=self.gc)

        feat_d = 4096
        if self.lc:
            feat_d += 128
        if self.gc:
            feat_d += 128
        # Fix the layers before conv3:
        for layer in range(5):
            for p in self.RCNN_base1[layer].parameters():
                p.requires_grad = False
        for layer in range(5):
            for p in self.RCNN_base2[layer].parameters():
                p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
