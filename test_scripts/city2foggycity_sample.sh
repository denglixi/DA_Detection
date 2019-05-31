#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python app/test_net_global_local.py --cuda --net vgg16 \
    --dataset foggy_cityscape --gc --lc --load_name $2 \
    --nw 5 \
    --test_cache
