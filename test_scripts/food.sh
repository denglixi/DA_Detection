#!/bin/sh

# test collected canteen
DATASET=foodArtsmt10

# test excl canteen
# DATASET=foodexclScience

CUDA_VISIBLE_DEVICES=$1 python ./app/test_net_global_local.py --cuda --net prefood --dataset $DATASET --load_name $2 \
    --nw 10 \
    --lc
    #--test_cache
    #--gc --lc  \
