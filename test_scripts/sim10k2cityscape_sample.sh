#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python ./app/test_net_multiscale.py \
    --cuda --net vgg16 \
    --dataset cityscape_car \
    --load_name $2
    #--gc --lc \
