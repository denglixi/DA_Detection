#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python ./app/test_net_global_local.py --cuda --net prefood --dataset foodArts_exclArts --gc --lc --load_name $2 \
    --nw 10
    #--test_cache \
