#!/bin/sh



CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_multiscale.py \
    --cuda --net vgg16_multiscale \
    --train_domain_loss \
    --dataset cityscape --dataset_t foggy_cityscape \
    --save_dir models \
    --nw 5
    #--gc --lc \
