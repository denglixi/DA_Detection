#!/bin/sh
NW=10
#{ foodexclArtsmt10 }
#DATASET=foodexclUTownmt10
DATASET=foodArtsmt10_few1
DATASET_T=foodUTown

# Resume
RESUME=true

# weather train domain

LOAD_NAME='./models/prefood/foodexclArtsmt10/a.pth'


CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local.py --cuda --net prefood \
    --r RESUME --load_name $LOAD_NAME \
    --nw $NW \
    --dataset $DATASET --dataset_t $DATASET_T \
    --gc --lc --save_dir $2 \
