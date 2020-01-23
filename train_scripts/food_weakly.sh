#! /bin/sh
#
# food_weakly.sh
# Copyright (C) 2020 lxdeng <lxdeng@next-gpu1>
#
# Distributed under terms of the MIT license.
#



NW=0
#{ foodexclArtsmt10 }
DATASET=foodexclArtsmt10
DATASET_T=foodArts

#trainval_net_no_cross
#trainval_net_global_local

DECAY_SETP=5


# !! train_domain_loss is necessary for training the domain align

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_nonFL.py --cuda --net res101 \
   --dataset $DATASET --dataset_t $DATASET_T \
   --save_dir $2 \
   --nw $NW \
   --gc --lc \
   --lr_decay_step=$DECAY_SETP \
   --train_domain_loss
   # --fixed_layer=4 \
