#!/bin/sh
NW=5
#{ foodexclArtsmt10 }
DATASET=foodexclUTownmt10
DATASET_T=foodUTown
CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local.py --cuda --net prefood \
    --dataset $DATASET --dataset_t $DATASET_T \
    --gc --lc --save_dir $2 \
