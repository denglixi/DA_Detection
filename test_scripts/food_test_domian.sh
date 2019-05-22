#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python ./app/test_domain_clser.py --cuda --net res101 --dataset foodArts_exclArts --gc --lc --load_name $2 \
    --nw 10
    #--test_cache \
