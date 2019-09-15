#!/bin/sh

NET=vgg16_multiscale

# source domain
# val.txt and images are not exist
DATASET=cityscape

# target domain
# DATASET=foggy_cityscape

CUDA_VISIBLE_DEVICES=$1 python app/test_net_multiscale.py --cuda --net $NET \
    --dataset $DATASET \
    --load_name $2 \
    --nw 5 \
    #--gc --lc \
