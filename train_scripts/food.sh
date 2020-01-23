#!/bin/sh
NW=8
#{ foodexclArtsmt10 }
DATASET=foodexclArtsmt10
DATASET_T=foodArts

#trainval_net_no_cross
#trainval_net_global_local

DECAY_SETP=5


# !! train_domain_loss is necessary for training the domain align

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_weakly_backbone.py --cuda --net res101 \
   --dataset $DATASET --dataset_t $DATASET_T \
   --save_dir $2 \
   --nw $NW \
   --bce_alpha 3 \
   --lr_decay_step=$DECAY_SETP \
   --train_domain_loss \
   --weakly_type sum
   # --fixed_layer=4 \
   #--gc --lc \
