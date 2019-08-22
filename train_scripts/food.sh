#!/bin/sh
NW=0
#{ foodexclArtsmt10 }
DATASET=foodexclArtsmt10
DATASET_T=foodArts

#trainval_net_no_cross
#trainval_net_global_local

DECAY_SETP=5


# !! train_domain_loss is necessary for training the domain align

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local.py --cuda --net res101 \
   --dataset $DATASET --dataset_t $DATASET_T \
   --save_dir $2 \
   --nw $NW \
   --gc --lc \
   --lr_decay_step=$DECAY_SETP \
   --train_domain_loss
   # --fixed_layer=4 \
