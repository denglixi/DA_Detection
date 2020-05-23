# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import pprint
import pdb
import time
import _init_paths

from torch.autograd import Variable
import torch.nn as nn
import torch
from torch import autograd

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss, \
    sampler_given_random_order
from model.utils.parser_func import parse_args, set_dataset_args
from model.faster_rcnn.faster_rcnn_weakly_backbone import FasterRCNN_Weakly
from model.faster_rcnn.faster_rcnn_weakly_multiscale_backbone import FasterRCNN_MultiWeakly
from model.faster_rcnn.faster_rcnn_teacher_student import FasterRCNN_Teacher_Student
# def set_imdb_args(args):
#    args.imdb_name = args.dataset + '_trainmt10'
#    args.imdb_name_target = args.dataset_t + '_train'
#    return args




def set_args(args):
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                     'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]',
                            'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


def settings(args):

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    # source support:
    # target support: food_[CollectedCateen]
    args = set_dataset_args(args)
    # args = set_food_imdb_name(args)
    set_args(args)
    settings(args)

    output_dir = 'CheckPoints/' + args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(
        args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    #args.batch_size = 2
    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)
    #sampler_batch_t2 = sampler(train_size_t, args.batch_size)

    if not args.fine_tune_on_target:
        dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                                imdb.num_classes, training=True)
        dataloader_s = torch.utils.data.DataLoader(dataset_s,
                                                batch_size=args.batch_size,
                                                sampler=sampler_batch,
                                                num_workers=args.num_workers)
    else:
        dataloader_s = None
        print('fine tune on target dataset')

    dataset_t = roibatchLoader(roidb_t, ratio_list_t,
                               ratio_index_t, args.batch_size,
                               imdb.num_classes, training=True,
                               transform4ts=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t,
                                               batch_size=args.batch_size,
                                               sampler=sampler_batch_t,
                                               num_workers=args.num_workers)

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

    # initilize the network here.

    fasterRCNN = FasterRCNN_Teacher_Student(imdb.classes,
                                   class_agnostic=args.class_agnostic,
                                   lc=args.lc, gc=args.gc,
                                   backbone_type='res101',
                                   pretrained=True,
                                   weakly_type=args.weakly_type,
                                   is_uda=args.train_uda_loss,
                                   is_img_wda=args.train_img_wda_loss,
                                   is_region_wda=args.train_region_wda_loss)
    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr,
                            'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        print("loading checkpoint %s" % (args.load_name))
        checkpoint = torch.load(args.load_name)
        fasterRCNN.load_state_dict(checkpoint['model'])


        if args.fine_tune_on_target:
            print("fine_tune...")
        else:
            print("resuming....")
            args.session = checkpoint['session']
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(args.checkpoint_interval / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")
    count_iter = 0
    with autograd.detect_anomaly():
        for epoch in range(args.start_epoch, args.max_epochs + 1):
            # setting to train mode
            fasterRCNN.train()
            loss_temp = 0
            start = time.time()
            if epoch % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma

            if args.fine_tune_on_target:
                dataloader_s = dataloader_t

            data_iter_s = iter(dataloader_s)
            data_iter_t = iter(dataloader_t)



            for step in range(iters_per_epoch):

                # each step: one source iteration and one target iteration
                # source iteration
                try:
                    data_s = next(data_iter_s)
                except:
                    data_iter_s = iter(dataloader_s)
                    data_s = next(data_iter_s)

                # eta = 1.0
                count_iter += 1
                # put source data into variable
                with torch.no_grad():
                    im_data.resize_(data_s[0].size()).copy_(data_s[0])
                    im_info.resize_(data_s[1].size()).copy_(data_s[1])
                    gt_boxes.resize_(data_s[2].size()).copy_(data_s[2])
                    num_boxes.resize_(data_s[3].size()).copy_(data_s[3])

                fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label, out_d_pixel_s, out_d_s = fasterRCNN(
                        im_data, im_info, gt_boxes, num_boxes)
                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                    + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                loss_temp += loss.item()

                ############################################################
                # target iteration and domain loss
                ##########################################################
                if args.train_uda_loss or args.train_img_wda_loss or args.train_region_wda_loss:


                    try:
                        data_t = next(data_iter_t)
                    except:
                        data_iter_t = iter(dataloader_t)
                        data_t = next(data_iter_t)

                    with torch.no_grad():
                        # put target data into variable
                        im_data.resize_(data_t[0].size()).copy_(data_t[0])
                        im_info.resize_(data_t[1].size()).copy_(data_t[1])
                        # gt is empty
                        gt_boxes.resize_(data_t[2].size()).copy_(data_t[2])
                        num_boxes.resize_(data_t[3].size()).copy_(data_t[3])
                        #gt_boxes.data.resize_(1, 1, 5).zero_()
                        # num_boxes.data.resize_(1).zero_()
                    out_d_pixel, out_d, img_bce_loss, region_bce_loss, weakly_consistency_loss = fasterRCNN(
                        im_data, im_info, gt_boxes, num_boxes, target=True)

                    if args.train_uda_loss:
                        # source domain label
                        domain_s = Variable(torch.zeros(out_d_s.size(0)).long().cuda())
                        # global alignment loss
                        dloss_s = 0.5 * FL(out_d_s, domain_s)
                        # local alignment loss
                        dloss_s_p = 0.5 * torch.mean(out_d_pixel_s ** 2)

                        # target domain label
                        domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
                        # global alignment loss
                        dloss_t = 0.5 * FL(out_d, domain_t)
                        # local alignment loss
                        dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
                        if args.dataset == 'sim10k':
                            loss += (dloss_s + dloss_t +
                                     dloss_s_p + dloss_t_p) * args.eta
                        loss += (dloss_s + dloss_t +
                                    dloss_s_p + dloss_t_p)  # * 10
                    if args.train_img_wda_loss:
                        loss += img_bce_loss * args.bce_alpha
                    if args.train_region_wda_loss:
                        loss += region_bce_loss * args.bce_alpha
                    loss += weakly_consistency_loss

                optimizer.zero_grad()
                with autograd.detect_anomaly():
                    loss.backward()
                optimizer.step()

                if step % args.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= (args.disp_interval + 1)

                    if args.mGPUs:
                        loss_rpn_cls = rpn_loss_cls.mean().item()
                        loss_rpn_box = rpn_loss_box.mean().item()
                        loss_rcnn_cls = RCNN_loss_cls.mean().item()
                        loss_rcnn_box = RCNN_loss_bbox.mean().item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        if args.train_uda_loss:
                            dloss_s = dloss_s.item()
                            dloss_t = dloss_t.item()
                            dloss_s_p = dloss_s_p.item()
                            dloss_t_p = dloss_t_p.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                          % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                    print("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                          (fg_cnt, bg_cnt, end - start))

                    output_str = "\t\t\trpn_cls: {:.4f}, rpn_box: {:.4f}, rcnn_cls: {:.4f}, rcnn_box {:.4f} ".format(
                        loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box)

                    if args.train_uda_loss:
                        output_str += "dloss s: {:.4f} dloss t: {:.4f} dloss s pixel: {:.4f} dloss t pixel: {:.4f} eta: {:.4f} ".format(
                            dloss_s, dloss_t, dloss_s_p, dloss_t_p,
                           args.eta)
                    if args.train_region_wda_loss:
                        output_str += "bce: %.4f " % (region_bce_loss)
                    if args.train_img_wda_loss:
                        output_str += "img bce: %.4f " % (img_bce_loss)

                    output_str += "weakly consistency loss: %.4f" % (weakly_consistency_loss)
                    print(output_str)
                    if args.use_tfboard:
                        info = {
                            'loss': loss_temp,
                            'loss_rpn_cls': loss_rpn_cls,
                            'loss_rpn_box': loss_rpn_box,
                            'loss_rcnn_cls': loss_rcnn_cls,
                            'loss_rcnn_box': loss_rcnn_box
                        }
                        logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                           (epoch - 1) * iters_per_epoch + step)

                    loss_temp = 0
                    start = time.time()
            save_name = os.path.join(output_dir,
                                     'globallocal_target_{}_eta_{}_local_context_{}_global_context_{}_gamma_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                         args.dataset_t, args.eta,
                                         args.lc, args.gc, args.gamma,
                                         args.session, epoch,
                                         step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
