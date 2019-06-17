#!/bin/sh

NET=vgg16_weakly

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local_weakly.py \
    --cuda --net $NET \
    --train_domain_loss \
    --dataset cityscape --dataset_t foggy_cityscape \
    --gc --lc --save_dir models \
    #--nw 4
