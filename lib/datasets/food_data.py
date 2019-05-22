from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval
from .voc_eval import loc_cls_eval, loc_cls_eval_for_image
from .voc_eval import rec_pre_eval_for_image_topk, rec_pre_eval_for_image_hierarchy
from .voc_eval import topk_acc_of_cls_per_dish
from .voc_eval import topk_acc_of_cls_per_dish_2
from .voc_eval import topk_falsealarm_of_cls_per_dish
from .food_category import get_categories

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class food_merge_imdb(imdb):
    def __init__(self, image_set, cantee, categories, devkit_path=None):
        """
        categories: All_trainval, exclYIH_trainval, ...
        """
        imdb.__init__(self, 'food_' + cantee + '_' + image_set)
        self._cantee = cantee
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(
            self._devkit_path, 'Food_' + self._cantee)
        self._classes = get_categories(categories)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._origin_img_len = len(self._image_index)
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'
        # init annopath, imagesetfile, cachedir
        self.init_evaluate_inform()
        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def init_evaluate_inform(self):
        self.annopath = os.path.join(
            self._devkit_path,
            'Food_' + self._cantee,
            'Annotations',
            '{:s}.xml')
        self.imagesetfile = os.path.join(
            self._devkit_path,
            'Food_' + self._cantee,
            'ImageSets',
            self._image_set + '.txt')
        self.cachedir = os.path.join(self._devkit_path, 'annotations_cache')

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'Food')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        # exlcude unused cls

        ori_num_objs = len(objs)
        num_objs = 0
        for obj in objs:
            try:
                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                num_objs += 1
            except:
                continue
        # assert num_objs == 0
        if num_objs == 0:
            import pdb
            pdb.set_trace()

        num_objs = num_objs  # len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        ix = 0
        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            # the range of food label is (0, width) which may cause by bugs in labelimg 1.4
            x1 = max(0.0, float(bbox.find('xmin').text) - 1)
            y1 = max(0.0, float(bbox.find('ymin').text) - 1)
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)

            try:
                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                # cls = int(obj.find('name').text.strip())
            except:
                print("Warning:*******cls can not found in file:******")
                print(filename)
                continue

                raise
            ishards[ix] = difficult
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            try:
                overlaps[ix, cls] = 1.0
            except:
                print(filename)
                raise
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'rotated': 0,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + \
            self._image_set + '_{:s}.txt'
        filedir = os.path.join(
            self._devkit_path, 'results', 'food' + self._cantee)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            #print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'Food_' + self._cantee,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'Food_' + self._cantee,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = False
        #print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            #print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        #print('~~~~~~~~')
        #print('Results:')
        #for ap in aps:
        #    print('{:.3f}'.format(ap))
        #print('{:.3f}'.format(np.mean(aps)))
        #print('~~~~~~~~')
        #print('')
        #print('--------------------------------------------------------------')
        #print('Results computed with the **unofficial** Python eval code.')
        #print('Results should be very close to the official MATLAB eval code.')
        #print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        #print('-- Thanks, The Management')
        #print('--------------------------------------------------------------')

        return zip(self._classes[1:], aps), np.mean(aps)

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_cls_loc_for_image(self, all_boxes, threshold=0.5):

        # gt of cls
        annopath = os.path.join(
            self._devkit_path,
            'Food_' + self._cantee,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'Food_' + self._cantee,
            'ImageSets',
            self._image_set + '.txt')

        cachedir = os.path.join(self._devkit_path, 'annotations_cache')

        return loc_cls_eval_for_image(
            all_boxes, annopath, imagesetfile, self._classes,  cachedir, threshold, 0.5)

    def evaluate_rec_pre_for_image_topk(self, all_boxes, classes, topk):
        annopath = self.annopath
        imagesetfile = self.imagesetfile
        cachedir = self.cachedir
        return rec_pre_eval_for_image_topk(all_boxes, annopath, imagesetfile, classes, cachedir, threshold=0.5, ovthresh=0.5, k=topk)

    def evaluate_hierarchy(self, all_boxes, classes):
        annopath = self.annopath
        imagesetfile = self.imagesetfile
        cachedir = self.cachedir
        return rec_pre_eval_for_image_hierarchy(all_boxes, annopath, imagesetfile, classes, cachedir, threshold=0.5, ovthresh=0.5, k=None)

    def evalute_topk_falsealarm(self, all_boxes, topk):
        # gt of cls
        annopath = self.annopath
        imagesetfile = self.imagesetfile
        cachedir = self.cachedir

        all_clsify = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            else:
                cls_accuracy = topk_falsealarm_of_cls_per_dish(
                    all_boxes, annopath, imagesetfile, i, cls, cachedir, 0.5, 0.5, topk)
                all_clsify.append(cls_accuracy)
        return list(zip(self._classes[1:], all_clsify))

    def evalute_topk_acc(self, all_boxes, topk):
        # gt of cls
        annopath = self.annopath
        imagesetfile = self.imagesetfile
        cachedir = self.cachedir

        all_acc = []
        all_fls = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            else:
                accuracy, falsealarm = topk_acc_of_cls_per_dish_2(
                    all_boxes, annopath, imagesetfile, i, cls, cachedir, 0.5, 0.5, topk)
                all_acc.append(accuracy)
                all_fls.append(falsealarm)
        return list(zip(self._classes[1:], all_acc)), list(zip(self._classes[1:], all_fls))

    def evaluate_cls_loc(self, all_boxes, threshold=0.5):
        # gt of cls
        annopath = self.annopath
        imagesetfile = self.imagesetfile
        cachedir = self.cachedir

        all_loc = []
        all_clsify = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            else:
                loc_accuracy, cls_accuracy = loc_cls_eval(
                    all_boxes, annopath, imagesetfile, i, cls, cachedir, threshold, 0.5)
                all_loc.append(loc_accuracy)
                all_clsify.append(cls_accuracy)
        return list(zip(self._classes[1:], all_loc)), list(zip(self._classes[1:], all_clsify))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        result = self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)
        return result

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed

    embed()
