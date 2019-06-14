#!/bin/sh

# test collected canteen
# DATASET=foodYIHmt10

# test excl canteen
DATASET=foodexclYIH

CUDA_VISIBLE_DEVICES=$1 python ./app/test_net_global_local.py --cuda --net prefood --dataset $DATASET --gc --lc --load_name $2 \
    --nw 10 \
