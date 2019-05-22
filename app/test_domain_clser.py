# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pprint
import time
import _init_paths

import torch

from torch.autograd import Variable
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
#from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.parser_func import parse_args, set_dataset_args
from datasets.food_category import get_categories

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    # args = set_dataset_args(args, test=True)
    args.imdbval_name = 'food_Arts_train_exclArts_train'
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

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    # initilize the network here.
    from model.faster_rcnn.vgg16_global_local import vgg16
    from model.faster_rcnn.resnet_global_local import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True,
                           class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True,
                            class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)
    # elif args.net == 'res50':
    #  fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,context=args.context)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (args.load_name))
    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    thresh = 0.0

    save_name = args.load_name.split('/')[-1]
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=False, normalize=False, path_return=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    if(args.test_cache):
        with open(det_file, 'rb') as f:
            all_boxes = pickle.load(f)
    else:
        fasterRCNN.eval()
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        for i in range(num_images):

            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            # print(data[0].size())
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, d_pred, _ = fasterRCNN(
                    im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            d_pred = d_pred.data
            path = data[4]
            pdb.set_trace()
            if (d_pred > 0.5):
                TP += 1
        print(TP/num_images)


