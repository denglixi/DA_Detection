#!/bin/sh
NW=5
#{ foodexclArtsmt10 }
#DATASET=foodexclUTownmt10
DATASET=foodArtsmt10_few1
DATASET_T=foodUTown

# Resume

LOAD_NAME=''


CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local.py --cuda --net prefood \
    --resume --load_name \
    --dataset $DATASET --dataset_t $DATASET_T \
    --gc --lc --save_dir $2 \
