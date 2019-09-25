#!/bin/sh

# test collected canteen
DATASET=foodArtsmt10

# test excl canteen
# DATASET=foodexclArts

NET=prefood #{res101, prefood, res101_local_unreversed}
CUDA_VISIBLE_DEVICES=$1 python ./app/test_net_global_local.py --cuda --net $NET --dataset $DATASET --load_name $2 \
    --gc --lc  \
    --test_cache
    # --gc \
    # --nw 10 \
