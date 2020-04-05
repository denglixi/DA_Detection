# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.pascal_voc_cyclewater import pascal_voc_cyclewater
from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart
from datasets.sim10k import sim10k
from datasets.water import water
from datasets.clipart import clipart
from datasets.sim10k_cycle import sim10k_cycle
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape
from datasets.food_data import food_merge_imdb

from model.utils.config import cfg
__sets = {}

for split in ['train', 'trainval', 'val', 'test']:
    name = 'cityscape_{}'.format(split)
    __sets[name] = (lambda split=split: cityscape(split))
for split in ['train', 'trainval', 'val', 'test']:
    name = 'cityscape_car_{}'.format(split)
    __sets[name] = (lambda split=split: cityscape_car(split))
for split in ['train', 'trainval', 'test']:
    name = 'foggy_cityscape_{}'.format(split)
    __sets[name] = (lambda split=split: foggy_cityscape(split))
for split in ['train', 'val']:
    name = 'sim10k_{}'.format(split)
    __sets[name] = (lambda split=split: sim10k(split))
for split in ['train', 'val']:
    name = 'sim10k_cycle_{}'.format(split)
    __sets[name] = (lambda split=split: sim10k_cycle(split))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_water_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split,
                        year=year: pascal_voc_water(split, year))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_cycleclipart_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split,
                        year=year: pascal_voc_cycleclipart(split, year))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_cyclewater_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split,
                        year=year: pascal_voc_cyclewater(split, year))
for year in ['2007']:
    for split in ['trainval', 'test']:
        name = 'clipart_{}'.format(split)
        __sets[name] = (lambda split=split: clipart(split, year))
for year in ['2007']:
    for split in ['train', 'test']:
        name = 'water_{}'.format(split)
        __sets[name] = (lambda split=split: water(split, year))


# Set up food_<canteen>_<split>_<trainingcategories>
splits = ['train', 'val', 'trainval', 'inner', 'test']
mt_splits = []
for n in [0, 10, 30, 50, 100]:
    for s in splits:
        mt_splits += [s+"mt{}".format(n)]
splits += mt_splits

# split inner dishes to train and val splits
innersplit = []
for sp in ['val', 'test']:
    for m in [10, 30, 50]:
        innersplit.append('innermt{}{}'.format(m, sp))
splits += innersplit

# for few shot fine tune in cross domain
# base on the origin val.txt
based_split = 'val'
innerfew = []
for sp in ['train', 'val']:
    for m in [10]:
        for few in [1, 5]:
            innerfew.append('innermt{}{}few{}mt{}{}'.format(
                m, based_split, few, m, sp))
splits += innerfew

# take few sample in inner between dataset of canteen and dataset of excl canteen as training data. And regard the lefts as validation.
inner_few = []
for fewN in [0, 1, 3, 5, 10]:
    for mtN in [10]:
        for d in ['train', 'val', 'test']:
            inner_few += ["innerfew{}mt{}{}".format(fewN, mtN, d)]
splits += inner_few

for cantee in ['exclYIH', "All", "exclArts", "exclUTown", "Science", "exclScience", "exclTechChicken", "exclTechMixedVeg", "YIH", "Arts", "TechChicken", "TechMixedVeg", "UTown", "EconomicBeeHoon"]:
    for split in splits:
        for category in ['exclYIH', "All", "exclArts", "exclUTown", "Science", "exclScience", "exclTechChicken", "exclTechMixedVeg", "YIH", "Arts", "TechChicken", "TechMixedVeg", "UTown", "EconomicBeeHoon"]:
            category_train = category + '_train'
            name = 'food_{}_{}_{}'.format(cantee, split, category_train)
            __sets[name] = (lambda split=split,
                            cantee=cantee, category_train=category_train: food_merge_imdb(split, cantee, category_train))
            __sets[name+'Fake'] = (lambda split=split,
                                   cantee=cantee,
                                   category_train=category_train: food_merge_imdb(split, cantee, category_train, devkit_path=os.path.join(cfg.DATA_DIR, 'FoodFake'), is_fake=True))
            for n in [10, 30, 50, 100]:
                category_mt10 = category + '_train_mt{}'.format(n)
                name = 'food_{}_{}_{}'.format(cantee, split, category_mt10)
                __sets[name] = (lambda split=split,
                                cantee=cantee, category_mt10=category_mt10: food_merge_imdb(split, cantee, category_mt10))
                __sets[name+'Fake'] = (lambda split=split,
                                   cantee=cantee,
                                   category_mt10=category_mt10: food_merge_imdb(split, cantee, category_mt10, devkit_path=os.path.join(cfg.DATA_DIR, 'FoodFake'), is_fake=True))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
