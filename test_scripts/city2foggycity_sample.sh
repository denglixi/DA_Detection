#!/bin/sh

NET=vgg16_weakly

# source domain
# val.txt and images are not exist
DATASET=cityscape

# target domain
# DATASET=foggy_cityscape

CUDA_VISIBLE_DEVICES=$1 python app/test_net_global_local.py --cuda --net $NET \
    --dataset $DATASET --gc --lc --load_name $2 \
    --nw 5 \
