#!/bin/sh
NW=8
#{ foodexclArtsmt10 }
DATASET=foodexclArtsmt10
DATASET_T=foodArts

#trainval_net_no_cross
#trainval_net_global_local

RESUME=False
#
LOAD_NAME='./res101_weakly_sum/res101/foodexclArtsmt10/globallocal_target_foodArts_eta_0.1_local_context_False_global_context_False_gamma_5_session_1_epoch_10_step_9999.pth'

DECAY_SETP=5


# !! train_domain_loss is necessary for training the domain align

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_weakly_backbone.py --cuda --net res101 \
   --dataset $DATASET --dataset_t $DATASET_T \
   --save_dir $2 \
   --nw $NW \
   --bce_alpha 3 \
   --lr_decay_step=$DECAY_SETP \
   --train_domain_loss \
   --weakly_type max \
    #--r $RESUME --load_name $LOAD_NAME \
   # --fixed_layer=4 \
   #--gc --lc \
