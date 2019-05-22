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

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.food import food
from datasets.food_data import food_merge_imdb
from datasets.school_lunch import school_lunch
from datasets.food_meta_data import food_meta_imdb

__sets = {}

for canteen in ["Arts"]:
    for split in ['train', 'test']:
        name = 'food_meta_{}_{}'.format(canteen, split)
        categories = "{}_trainval".format(canteen)
        __sets[name] = (lambda split=split, canteen=canteen:
                        food_meta_imdb(split, canteen, categories))

# Set up food_<canteen>_<split>_<trainingcategories>
splits = ['train', 'val', 'trainval', 'inner', 'test']
mt_splits = []
for n in [0, 10, 30, 50, 100]:
    for s in splits:
        mt_splits += [s+"mt{}".format(n)]
splits += mt_splits

innersplit = []
for sp in ['val', 'test']:
    for m in [10, 30, 50]:
        innersplit.append('innermt{}{}'.format(m, sp))

splits += innersplit

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
            for n in [10, 30, 50, 100]:
                category_mt10 = category + '_train_mt{}'.format(n)
                name = 'food_{}_{}_{}'.format(cantee, split, category_mt10)
                __sets[name] = (lambda split=split,
                                cantee=cantee, category_mt10=category_mt10: food_merge_imdb(split, cantee, category_mt10))

#__sets["Food_EconomicBeeHoon_train"] = food_meta_imdb(train, )

# Set up school lunch
for split in ['train', 'val', 'trainval', 'test']:
    name = 'schoollunch_{}'.format(split)
    __sets[name] = (lambda split=split: school_lunch(split))
# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
    for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version, split)
        __sets[name] = (lambda split=split,
                        version=version: vg(version, split))

# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path,
                    data_path=data_path: imagenet(split, devkit_path, data_path))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
