#!/bin/sh

# test collected canteen
DATASET=foodSciencemt10

# test excl canteen
# DATASET=foodexclScience

CUDA_VISIBLE_DEVICES=$1 python ./app/test_net_global_local.py --cuda --net prefood --dataset $DATASET --load_name $2 \
    --nw 10 \
    --lc --gc
    #--test_cache
    #--gc --lc  \
