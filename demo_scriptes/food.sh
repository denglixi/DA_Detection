#!/bin/sh

# test collected canteen
#DATASET=foodTechChickenmt10

# test excl canteen
DATASET=foodexclTechChicken

CUDA_VISIBLE_DEVICES=$1 python ./app/test_net_global_local.py --cuda --net prefood --dataset $DATASET --gc --lc --load_name $2 \
    --nw 10 \
    #--test_cache
