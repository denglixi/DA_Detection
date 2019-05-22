# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from .id2name import id2chn
from .sub2main import sub2main_dict


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False, classname="Object AP", draw_ap=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        if draw_ap:
            zh_font = FontProperties(fname='./simsun.ttc')
            mrec_i = np.concatenate((mrec[i], [1.]))
            mpre_i = np.concatenate((mpre[i], [0.]))
            plt.figure()
            plt.title(id2chn[classname], fontproperties=zh_font)
            plt.plot(mrec_i, mpre_i)
            plt.show()

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def cal_overlap(boxes_array, box):
    ixmin = np.maximum(boxes_array[:, 0], box[0])
    iymin = np.maximum(boxes_array[:, 1], box[1])
    ixmax = np.minimum(boxes_array[:, 2], box[2])
    iymax = np.minimum(boxes_array[:, 3], box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
           (boxes_array[:, 2] - boxes_array[:, 0] + 1.) *
           (boxes_array[:, 3] - boxes_array[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return ovmax, jmax


def get_gt_recs(cachedir, imagesetfile, annopath):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annotations
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except Exception:
                recs = pickle.load(f, encoding='bytes')
    return imagenames, recs


def rec_pre_eval_for_image_hierarchy(all_boxes,
                                     annopath,
                                     imagesetfile,
                                     classes,
                                     cachedir,
                                     threshold,
                                     ovthresh,
                                     k):

    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    recall_all = []
    accuracy_all = []
    for img_idx, imagename in enumerate(imagenames):
        TP = 0
        FP = 0
        # get gt of image
        img_gt_recs = recs[imagename]
        npos_img = len(img_gt_recs)
        if npos_img == 0:
            continue

        if k is None:
            topk = npos_img
        else:
            topk = k
        # get det result of image [ [boxes of class1], ...[ boxes of clsN ] ]
        img_all_boxes = [b[img_idx] for b in all_boxes]

        # add cls_idx to box_cls

        bboxes = None

        for cls_idx, boxes_of_cls in enumerate(img_all_boxes):
            if len(boxes_of_cls) != 0:
                cls_cloumn = np.zeros(len(boxes_of_cls)) + cls_idx
                # img_all_boxes[cls_idx] = np.c_[boxes_of_cls, cls_cloumn]
                if bboxes is None:
                    bboxes = np.c_[boxes_of_cls, cls_cloumn]
                else:
                    bboxes = np.vstack(
                        (bboxes, np.c_[boxes_of_cls, cls_cloumn]))
        if bboxes is None:
            recall = 0
            accuracy = 0
        else:
            bboxes = bboxes[bboxes[:, 4].argsort()]
            bboxes = bboxes[::-1]
            for det_box in bboxes[:topk]:
                cls_name = classes[int(det_box[5])]
                # if obj['name'] == cls_name]
                R = [obj for obj in recs[imagename]]
                bbox = np.array([x['bbox'] for x in R])
                if len(bbox) > 0:
                    omax, jmax = cal_overlap(bbox, det_box)
                    if omax > ovthresh:
                        if cls_name == R[jmax]['name']:
                            TP += 1
                        elif sub2main_dict[cls_name] == sub2main_dict[R[jmax]['name']]:
                            TP += 0.5
                            FP += 0.5
                        else:
                            FP += 1
                    else:
                        FP += 1
                else:
                    FP += 1

            recall = TP / np.float32(npos_img)

            if TP + FP == 0:
                accuracy = 0
            else:
                accuracy = TP / np.float32(TP+FP)

        recall_all.append(recall)
        accuracy_all.append(accuracy)

    # use np to return nan while the npos is zero
    return np.mean(recall_all), np.mean(accuracy_all)


def rec_pre_eval_for_image_topk_back(all_boxes,
                                     annopath,
                                     imagesetfile,
                                     classes,
                                     cachedir,
                                     threshold,
                                     ovthresh,
                                     k):

    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    recall_all = []
    accuracy_all = []
    for img_idx, imagename in enumerate(imagenames):
        TP = 0
        FP = 0
        # get gt of image
        img_gt_recs = recs[imagename]

        npos_img = len(img_gt_recs)

        # exlucde rice('1') from npos_img
        for rec in img_gt_recs:
            if rec['name'] == '1':
                npos_img -= 1
        if npos_img == 0:
            continue

        if k is None:
            topk = npos_img
        else:
            topk = k
        # get det result of image [ [boxes of class1], ...[ boxes of clsN ] ]
        img_all_boxes = [b[img_idx] for b in all_boxes]

        # add cls_idx to box_cls

        bboxes = None

        for cls_idx, boxes_of_cls in enumerate(img_all_boxes):
            if len(boxes_of_cls) != 0:
                cls_cloumn = np.zeros(len(boxes_of_cls)) + cls_idx
                # img_all_boxes[cls_idx] = np.c_[boxes_of_cls, cls_cloumn]
                if bboxes is None:
                    bboxes = np.c_[boxes_of_cls, cls_cloumn]
                else:
                    bboxes = np.vstack(
                        (bboxes, np.c_[boxes_of_cls, cls_cloumn]))
        if bboxes is None:
            recall = 0
            accuracy = 0
        else:
            bboxes = bboxes[bboxes[:, 4].argsort()]
            bboxes = bboxes[::-1]
            topn_used = 0
            for det_box in bboxes:
                if topn_used == topk:
                    break
                cls_name = classes[int(det_box[5])]
                # exlucde rice('1') from npos_img
                if cls_name == '1':
                    continue
                topn_used += 1
                R = [obj for obj in recs[imagename] if obj['name'] == cls_name]
                bbox = np.array([x['bbox'] for x in R])
                if len(bbox) > 0:
                    omax, jmax = cal_overlap(bbox, det_box)
                    if omax > ovthresh:
                        TP += 1
                    else:
                        FP += 1
                else:
                    FP += 1

            recall = TP / np.float32(npos_img)

            if TP + FP == 0:
                accuracy = 0
            else:
                accuracy = TP / np.float32(TP+FP)

        recall_all.append(recall)
        accuracy_all.append(accuracy)

    # use np to return nan while the npos is zero
    return np.mean(recall_all), np.mean(accuracy_all)


def rec_pre_eval_for_image_topk(all_boxes,
                                annopath,
                                imagesetfile,
                                classes,
                                cachedir,
                                threshold,
                                ovthresh,
                                k):

    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    recall_all = []
    accuracy_all = []

    for img_idx, imagename in enumerate(imagenames):
        TP = 0
        FP = 0
        # get gt of image
        try:
            img_gt_recs = recs[imagename]
        except:
            print(imagename)
            continue

        img_categories = []
        for recs_img in img_gt_recs:
            if recs_img['name'] not in img_categories:
                img_categories.append(recs_img['name'])

        npos_img = len(img_gt_recs)

        # exlucde rice('1') from npos_img
        if npos_img == 0:
            continue

        if k is None:
            topk = npos_img
        else:
            topk = k
        # get det result of image [ [boxes of class1], ...[ boxes of clsN ] ]
        img_all_boxes = [b[img_idx] for b in all_boxes]

        # add cls_idx to box_cls

        bboxes = None

        for cls_idx, boxes_of_cls in enumerate(img_all_boxes):
            if len(boxes_of_cls) != 0:

                # only access top 1 of each categories
                cls_cloumn = np.zeros(len(boxes_of_cls)) + cls_idx
                #cls_cloumn = cls_idx

                #boxes_of_cls =  boxes_of_cls[np.argmax(boxes_of_cls[:, 4])]
                if bboxes is None:
                    #i bboxes = np.r_[boxes_of_cls, cls_cloumn + np.zeros(1)]
                    bboxes = np.c_[boxes_of_cls, cls_cloumn ]
                else:
                    #bboxes = np.vstack(
                    #    (bboxes, np.r_[boxes_of_cls, cls_cloumn + np.zeros(1) ]))
                    bboxes = np.vstack(
                        (bboxes, np.c_[boxes_of_cls, cls_cloumn ]))


        if bboxes is None:
            recall = 0
            accuracy = 0
        else:
            bboxes = bboxes[bboxes[:, 4].argsort()]
            bboxes = bboxes[::-1]
            topn_used = 0
            for det_box in bboxes:
                if topn_used == topk:
                    break
                cls_name = classes[int(det_box[5])]
                topn_used += 1
                if str(cls_name) in img_categories:
                    TP += 1
                else:
                    FP += 1
                #R = [obj for obj in recs[imagename] if obj['name'] == cls_name]
                #bbox = np.array([x['bbox'] for x in R])
                #if len(bbox) > 0:
                #    omax, jmax = cal_overlap(bbox, det_box)
                #    if omax > ovthresh:
                #        TP += 1
                #    else:
                #        FP += 1
                #else:
                #    FP += 1

            recall = TP / np.float32(len(img_categories))
            if recall > 1:
                recall = 1


            if TP + FP == 0:
                accuracy = 0
            else:
                accuracy = TP / np.float32(TP+FP)

        recall_all.append(recall)
        accuracy_all.append(accuracy)

    # use np to return nan while the npos is zero
    return np.mean(recall_all), np.mean(accuracy_all)


def loc_cls_eval_for_image(all_boxes,
                           annopath,
                           imagesetfile,
                           classes,
                           cachedir,
                           threshold,
                           ovthresh):
    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)
    recall_all = []
    accuracy_all = []
    for img_idx, imagename in enumerate(imagenames):
        img_gt_recs = recs[imagename]
        npos_img = len(img_gt_recs)
        if npos_img == 0:
            continue
        TP = 0
        FP = 0

        # all det boxes in one images
        img_all_boxes = [b[img_idx] for b in all_boxes]
        for cls_idx, det_boxes in enumerate(img_all_boxes):
            if len(det_boxes) > 0:
                cls_name = classes[cls_idx]
                R = [obj for obj in recs[imagename] if obj['name'] == cls_name]
                bbox = np.array([x['bbox'] for x in R])
                for det_box in det_boxes:
                    # filter the det box by score
                    if det_box[4] > threshold:
                        det_box = np.array(det_box)
                    else:
                        continue
                    #
                    if len(bbox) > 0:
                        omax, jmax = cal_overlap(bbox, det_box)
                        if omax > ovthresh:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        FP += 1

        recall = TP / np.float32(npos_img)
        if TP + FP == 0:
            accuracy = 0
        else:
            accuracy = TP / np.float32(TP+FP)

        recall_all.append(recall)
        accuracy_all.append(accuracy)

    # use np to return nan while the npos is zero
    return np.mean(recall_all), np.mean(accuracy_all)


def topk_acc_of_cls_per_dish_2(all_boxes,
                               annopath,
                               imagesetfile,
                               cls_idx,
                               classname,
                               cachedir,
                               threshold=0.5,
                               ovthresh=0.5,
                               topk=5
                               ):
    """topk_acc_of_cls_per_dish_2
    topk accuracy of the dish whose categrioy is classname
    topk accuracy: TP/npos. TP is 1 when topK reuslt has the correct result, otherwise is 0

    :param all_boxes:
    :param annopath:
    :param imagesetfile:
    :param classname:
    :param cachedir:
    :param threshold:
    :param ovthresh:
    :param topk:
    """

    # first load gt
    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    # extract gt objects for this class
    class_recs = {}
    cls_idx_recs = {}
    npos = 0
    for img_idx, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        cls_idx_recs[img_idx] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    TP = 0
    FP = 0
    # evaluate on each image
    for img_idx in cls_idx_recs:
        # get all box of img
        img_all_boxes = [b[img_idx] for b in all_boxes]
        # add cls_idx to box_cls
        bboxes = None
        for cls_idx_det, boxes_of_cls in enumerate(img_all_boxes):
            if len(boxes_of_cls) != 0:
                cls_cloumn = np.zeros(len(boxes_of_cls)) + cls_idx_det
                # img_all_boxes[cls_idx] = np.c_[boxes_of_cls, cls_cloumn]
                if bboxes is None:
                    bboxes = np.c_[boxes_of_cls, cls_cloumn]
                else:
                    bboxes = np.vstack(
                        (bboxes, np.c_[boxes_of_cls, cls_cloumn]))

        # choose topk results which include cls information
        if bboxes is None:
            continue
        else:
            # sort results
            bboxes = bboxes[bboxes[:, 4].argsort()]
            bboxes = bboxes[::-1]
            bboxes = bboxes[np.where(bboxes[:, 5] != 1)]

            # topk result
            for det_box in bboxes[:topk]:
                # cls_name = classes[int(det_box[5])]
                if int(det_box[5]) != cls_idx:
                    continue
                # gt
                bbox = cls_idx_recs[img_idx]['bbox']
                if len(bbox) > 0:
                    omax, jmax = cal_overlap(bbox, det_box)
                    if omax > ovthresh:
                        # if int(det_box[6]) == cls_idx
                        TP += 1
                    else:
                        FP += 1
                else:
                    FP += 1

    # use np to return nan while the npos is zero
    accuracy = TP / np.float32(npos)
    falseAlarm = FP / np.float32(TP + FP)

    return accuracy, falseAlarm


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


def topk_FP_mixed_matrix(all_boxes,
                         annopath,
                         imagesetfile,
                         cls_idx,
                         classname,
                         cachedir,
                         threshold=0.5,
                         ovthresh=0.5,
                         topk=5
                         ):

    # first load gt
    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    # find image without specified cls
    FN_image = []
    for img_idx, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        if len(R) > 0:
            continue
        else:
            FN_image.append(img_idx)

    FP = 0
    # evaluate on each image

    for img_idx in FN_image:
        bboxes = get_det_bbox_with_cls_of_img(all_boxes, img_idx)
        # choose topk results which include cls information
        if bboxes is None:
            continue
        else:
            # sort results
            bboxes = bboxes[bboxes[:, 4].argsort()]
            bboxes = bboxes[::-1]
            # delete rice
            bboxes = bboxes[np.where(bboxes[:, 5] != 1)]
            # topk result
            for det_box in bboxes[:topk]:
                # cls_name = classes[int(det_box[5])]
                if int(det_box[5]) != cls_idx:
                    continue
                else:
                    FP += 1
    # use np to return nan while the npos is zero
    falseAlarm = FP  # / np.float32(len(FN_image))
    return falseAlarm


def topk_falsealarm_of_cls_per_dish(all_boxes,
                                    annopath,
                                    imagesetfile,
                                    cls_idx,
                                    classname,
                                    cachedir,
                                    threshold=0.5,
                                    ovthresh=0.5,
                                    topk=5
                                    ):

    # first load gt
    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    # find image without specified cls
    FN_image = []
    for img_idx, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        if len(R) > 0:
            continue
        else:
            FN_image.append(img_idx)

    FP = 0
    # evaluate on each image

    for img_idx in FN_image:
        bboxes = get_det_bbox_with_cls_of_img(all_boxes, img_idx)
        # choose topk results which include cls information
        if bboxes is None:
            continue
        else:
            # sort results
            bboxes = bboxes[bboxes[:, 4].argsort()]
            bboxes = bboxes[::-1]
            # delete rice
            bboxes = bboxes[np.where(bboxes[:, 5] != 1)]
            # topk result
            for det_box in bboxes[:topk]:
                # cls_name = classes[int(det_box[5])]
                if int(det_box[5]) != cls_idx:
                    continue
                else:
                    FP += 1
    # use np to return nan while the npos is zero
    falseAlarm = FP  # / np.float32(len(FN_image))
    return falseAlarm


def topk_acc_of_cls_per_dish(all_boxes,
                             annopath,
                             imagesetfile,
                             cls_idx,
                             classname,
                             cachedir,
                             threshold=0.5,
                             ovthresh=0.5,
                             topk=5
                             ):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    # extract gt objects for this class
    class_recs = {}
    cls_idx_recs = {}
    npos = 0
    for img_idx, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        cls_idx_recs[img_idx] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    cls_TP = 0

    for img_idx in cls_idx_recs:
        img_i_cls_boxes = all_boxes[cls_idx][img_idx]
        # img_all_boxes = [b[img_idx] for b in all_boxes if len(b[img_idx]) > 0]
        if len(img_i_cls_boxes) == 0:
            continue

        # filter by threshold
        img_i_cls_boxes = img_i_cls_boxes[:topk]
        # correct loc
        BBGT = cls_idx_recs[img_idx]['bbox'].astype(float)
        if BBGT.size > 0:
            for bgt in BBGT:
                if img_i_cls_boxes.size > 0:
                    omax, jmax = cal_overlap(img_i_cls_boxes, bgt)
                    if omax > ovthresh:
                        cls_TP += 1

    # use np to return nan while the npos is zero
    accuracy = cls_TP / np.float32(npos)
    return accuracy


def loc_cls_eval(all_boxes,
                 annopath,
                 imagesetfile,
                 cls_idx,
                 classname,
                 cachedir,
                 threshold=0.5,
                 ovthresh=0.5,
                 ):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    # extract gt objects for this class
    class_recs = {}
    cls_idx_recs = {}
    npos = 0
    for img_idx, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        cls_idx_recs[img_idx] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    loc_TP = 0
    cls_TP = 0

    for img_idx in cls_idx_recs:

        img_i_cls_boxes = all_boxes[cls_idx][img_idx]
        img_all_boxes = [b[img_idx] for b in all_boxes if len(b[img_idx]) > 0]
        if len(img_all_boxes) > 0:
            img_i_boxes = np.concatenate(
                np.array(img_all_boxes))
        else:
            continue

        # filter by threshold
        img_i_boxes = img_i_boxes[np.where(img_i_boxes[:, 4] > threshold)]
        img_i_cls_boxes = img_i_cls_boxes[
            np.where(img_i_cls_boxes[:, 4] > threshold)]
        # correct loc
        BBGT = cls_idx_recs[img_idx]['bbox'].astype(float)
        if BBGT.size > 0:
            for bgt in BBGT:
                if img_i_boxes.size > 0:
                    omax, jmax = cal_overlap(img_i_boxes, bgt)
                    if omax > ovthresh:
                        loc_TP += 1
                if img_i_cls_boxes.size > 0:
                    omax, jmax = cal_overlap(img_i_cls_boxes, bgt)
                    if omax > ovthresh:
                        cls_TP += 1

    # use np to return nan while the npos is zero
    loc_accuracy = loc_TP / np.float32(npos)
    cls_accuracy = cls_TP / np.float32(npos)
    return loc_accuracy, cls_accuracy


def voc_eval_main_cls(detpath,
                      annopath,
                      imagesetfile,
                      classname,
                      cachedir,
                      ovthresh=0.5,
                      use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [" ".join(x[0:-5]) for x in splitlines]
    confidence = np.array([float(x[-5]) for x in splitlines])
    BB = np.array([[float(z) for z in x[-4:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric, classname)

    return rec, prec, ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    imagenames, recs = get_gt_recs(cachedir, imagesetfile, annopath)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [" ".join(x[0:-5]) for x in splitlines]
    confidence = np.array([float(x[-5]) for x in splitlines])
    BB = np.array([[float(z) for z in x[-4:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric, classname)

    return rec, prec, ap
