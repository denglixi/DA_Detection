#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 lixi <lixi@icube-next-02>
#
# Distributed under terms of the MIT license.

"""

"""
from .food_data import food_merge_imdb

class food_fake_imdb(food_merge_imdb):
    #
    #
    def __init__(self, image_set, cantee, categories, devkit_path=None):
        super(food_fake_imdb, self).__init__(image_set, cantee, categories)
