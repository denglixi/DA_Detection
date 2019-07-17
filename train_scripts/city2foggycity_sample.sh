#!/bin/sh


NET=vgg16

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local.py \
    --cuda --net $NET \
    --train_domain_loss \
    --dataset cityscape --dataset_t foggy_cityscape \
    --gc --lc --save_dir models \
    #--nw 5
