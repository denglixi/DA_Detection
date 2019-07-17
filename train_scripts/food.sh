#!/bin/sh
NW=0
#{ foodexclArtsmt10 }
DATASET=foodexclTechChickenmt10
DATASET_T=foodTechChicken
CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_local.py --cuda --net prefood \
    --dataset $DATASET --dataset_t $DATASET_T \
    --gc --lc --save_dir $2 \
    --nw $NW
