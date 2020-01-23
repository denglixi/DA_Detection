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
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.parser_func import parse_args, set_dataset_args
from datasets.food_category import get_categories
from model.faster_rcnn.vgg16_global_local import vgg16
from model.faster_rcnn.resnet_global_local import resnet
from model.faster_rcnn.prefood_res50_attention import PreResNet50Attention
from model.faster_rcnn.vgg16_global_local_weakly import vgg16_weakly
from model.faster_rcnn.resnet_global_local_weakly import resnet_weakly
from model.faster_rcnn.vgg16_global_local_weakly_sum import vgg16_weakly_sum
from model.faster_rcnn.resnet_global_local_unreversed import resnet_local_unreversed
from model.faster_rcnn.vgg16_multiscale import vgg16_multiscale
from model.faster_rcnn.faster_rcnn_global_local_backbone import FasterRCNN
import cv2
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

    # set dataset : public dataset or food

    # public dataset
    args = set_dataset_args(args, test=True)

    test_canteen = args.imdbval_name.split('_')[1]

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

    # if args.net == 'vgg16':
    #    fasterRCNN = vgg16(imdb.classes, pretrained=True,
    #                       class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)

    # elif args.net == 'vgg16_multiscale':
    #    fasterRCNN = vgg16_multiscale(imdb.classes, pretrained=False,
    #                                  class_agnostic=args.class_agnostic,
    #                                  lc=args.lc,
    #                                  gc=args.gc)

    # elif args.net == 'res101':
    #    fasterRCNN = resnet(imdb.classes, 101, pretrained=True,
    #                        class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)

    # elif args.net == 'res101_local_unreversed':
    #    fasterRCNN = resnet_local_unreversed(imdb.classes, 101, pretrained=True,
    #                                         class_agnostic=args.class_agnostic,
    #                                         lc=args.lc, gc=args.gc)

    # elif args.net == 'prefood':
    #    fasterRCNN = PreResNet50Attention(imdb.classes,  pretrained=True,
    #                                      class_agnostic=args.class_agnostic,
    #                                      lc=args.lc, gc=args.gc)

    # elif args.net == 'vgg16_weakly':
    #    fasterRCNN = vgg16_weakly(imdb.classes, pretrained=True,
    #                              class_agnostic=args.class_agnostic,
    #                              lc=args.lc,
    #                              gc=args.gc)

    # elif args.net == 'vgg16_weakly_sum':
    #    fasterRCNN = vgg16_weakly_sum(imdb.classes, pretrained=True,
    #                                  class_agnostic=args.class_agnostic,
    #                                  lc=args.lc,
    #                                  gc=args.gc)

    # elif args.net == 'res50':
    #  fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,context=args.context)
    # else:
    #    print("network is not defined")
    #    pdb.set_trace()

    fasterRCNN = FasterRCNN(imdb.classes, class_agnostic=args.class_agnostic,
                            lc=args.lc, gc=args.gc, backbone_type='res101')
    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (args.load_name))
    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
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
                                             shuffle=False, num_workers=args.num_workers,
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
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            fasterRCNN_result = fasterRCNN(
                im_data, im_info, gt_boxes, num_boxes)
            if len(fasterRCNN_result) == 10:
                # normal global local model
                rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label, d_pred, _ = fasterRCNN_result
            if len(fasterRCNN_result) == 9:
                # normal global or local model
                rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label, d_pred = fasterRCNN_result
            elif len(fasterRCNN_result) == 13:
                # vgg16_multiscale model
                rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label, d_pred, _, _1, _2, _3 = fasterRCNN_result

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            d_pred = d_pred.data
            path = data[4]

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
                            1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()

            vis = args.vis

            im2show = None
            if vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)

            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat(
                        (cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(
                            im2show, imdb.classes[j], np.array(cls_dets.cpu().numpy()), 0.5, [255, 0, 0])

                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(
                            all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'
                             .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            # To save image for analysis
            # Limit to threshhold detections *over all classes*
            if vis:
                threshold_of_vis = 0.1
                all_boxes_save_for_vis = all_boxes.copy()
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes_save_for_vis[j][i][:, -1]
                                              for j in xrange(1, imdb.num_classes)])
                    # np.sort(image_scores)[-max_per_image]
                    image_thresh = threshold_of_vis
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(
                            all_boxes_save_for_vis[j][i][:, -1] >= image_thresh)[0]
                        all_boxes_save_for_vis[j][i] = all_boxes_save_for_vis[j][i][keep, :]

                if all_boxes_save_for_vis[1][i].shape[0] == 0:
                    continue
                boxes_of_i = np.array([_[i]
                                       for _ in all_boxes_save_for_vis])

                # filter boxes with lower score
                # It is 0 for batch size is 1
                gt_boxes_cpu = gt_boxes.cpu().numpy()[0]
                try:
                    gt_boxes_cpu[:, 0:4] /= float(im_info[0][2].cpu().numpy())
                except:
                    pdb.set_trace()

                save_vis_root_path = './savevis/{}/{}/{}_{}_{}/'.format(args.net, args.imdbval_name,
                                                                        args.checksession, args.checkepoch, args.checkpoint)

                # show ground-truth
                for gt_b in gt_boxes_cpu:
                    im2show = vis_detections(
                        im2show, imdb.classes[int(gt_b[-1])], gt_b[np.newaxis, :], 0.1, (0, 255, 0), False)

                i_row, i_c, _ = im2show.shape
                im2show = cv2.resize(im2show, (int(i_c/2), int(i_row/2)))

                # save all
                save_vis_path = save_vis_root_path + \
                    'All/'
                if not os.path.exists(save_vis_path):
                    os.makedirs(save_vis_path)
                cv2.imwrite(os.path.join(save_vis_path,
                                         imdb.image_index[i]+'.jpg'), im2show)

                # save by condition
                # 1.gt未检测到
                # 2. gt类别错误(TODO)
                for gt_b in gt_boxes_cpu:
                    gt_cls_idx = int(gt_b[4])
                    # 1 && 2
                    if len(boxes_of_i[gt_cls_idx]) == 0:
                        save_vis_path = save_vis_root_path + \
                            'FN/' + imdb.classes[int(gt_cls_idx)]
                        if not os.path.exists(save_vis_path):
                            os.makedirs(save_vis_path)
                        # im2vis_analysis = vis_detections(
                        #    im2show, imdb.classes[int(gt_b[-1])], gt_b[np.newaxis,:], 0.1, (204, 0, 0))
                        cv2.imwrite(os.path.join(save_vis_path,
                                                 imdb.image_index[i]+'.jpg'), im2show)

                gt_classes = [int(_[-1]) for _ in gt_boxes_cpu]
                # 3. FP
                for bi, det_b_cls in enumerate(boxes_of_i):
                    if len(det_b_cls) > 0 and any(det_b_cls[:, 4] > 0.5):
                        if bi not in gt_classes:
                            save_vis_path = save_vis_root_path + \
                                'FP/' + str(imdb.classes[bi])
                            if not os.path.exists(save_vis_path):
                                os.makedirs(save_vis_path)
                            cv2.imwrite(os.path.join(save_vis_path,
                                                     imdb.image_index[i]+'.jpg'), im2show)

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print("test time: %0.4fs" % (end - start))

    print('Evaluating detections')
    # evaluate mAP
    cls_ap_zip, dataset_mAP = imdb.evaluate_detections(
        all_boxes, output_dir)
    cls_ap = list(cls_ap_zip)

    # for excl canteen
    if 'excl' in test_canteen:
        val_categories = get_categories(
            "{}".format(test_canteen)+"_"+"trainmt10")
    # for collcted canteen cross domain test, which is the inner split
    else:
        val_categories = get_categories("{}".format(test_canteen)+"_"+"inner")
    map_exist_cls = []
    if val_categories is not None:
        for cls, ap in cls_ap:
            if cls in val_categories:
                if np.isnan(ap):
                    continue
                else:
                    map_exist_cls.append(ap)
                    print(cls, ap)
        map_exist_cls = sum(map_exist_cls) / len(map_exist_cls)
        print(map_exist_cls)
    else:
        print(cls_ap_zip, dataset_mAP)

    save_record_file_path = "/".join(args.load_name.split('/')[:-1])
    load_model_name = args.load_name.split('/')[-1]
    with open(save_record_file_path + '/record.txt', 'a') as f:
        f.write(str(load_model_name) + '\t')
        f.write(str(map_exist_cls) + '\n')
