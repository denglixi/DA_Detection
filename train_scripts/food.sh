#!/bin/sh
NW=0
#{ foodexclArtsmt10 }
DATASET=foodexclSciencemt10
DATASET_T=foodScience

#trainval_net_no_cross
#trainval_net_global_local



# !! train_domain_loss is necessary for training the domain align

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_no_cross.py --cuda --net prefood \
    --dataset $DATASET --dataset_t $DATASET_T \
    --gc --lc --save_dir $2 \
    --nw $NW \
    --fixed_layer=4 \
    --train_domain_loss
