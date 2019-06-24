#!/bin/sh
NW=5
#{ foodexclArtsmt10 }
DATASET=foodexclSciencemt10
DATASET_T=foodScience
CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local.py --cuda --net prefood \
    --dataset $DATASET --dataset_t $DATASET_T \
    --gc --lc --save_dir $2 \
    --nw $NW
