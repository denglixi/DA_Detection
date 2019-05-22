#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local.py \
    --cuda --net vgg16 \
    --dataset cityscape --dataset_t foggy_cityscape \
    --gc --lc --save_dir models \
    --nw 5
