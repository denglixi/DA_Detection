#!/bin/sh

# test collected canteen
DATASET=foodArtsmt10

# test excl canteen
# DATASET=foodexclArts

NET=res101 #{res101, prefood}
CUDA_VISIBLE_DEVICES=$1 python ./app/test_net_global_local.py --cuda --net $NET --dataset $DATASET --load_name $2 \
    --gc --lc  \
    --nw 10 \
    #--test_cache
    #--gc \
