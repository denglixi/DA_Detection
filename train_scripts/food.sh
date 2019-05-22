#!/bin/sh
NW=5
#{ foodexclArtsmt10 }
DATASET=foodexclYIHmt10
DATASET_T=foodYIH
CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local.py --cuda --net prefood \
    --dataset $DATASET --dataset_t $DATASET_T \
    --gc --lc --save_dir $2 \
