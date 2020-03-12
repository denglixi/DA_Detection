#!/bin/sh
NW=8
#{ foodexclArtsmt10 }
DATASET=foodexclArtsmt10
DATASET_T=foodArtsmt10

#DATASET=foodexclUTownmt10
#DATASET_T=foodUTownmt10

#trainval_net_no_cross
#trainval_net_global_local

#DATASET=pascal_voc_0712
#DATASET_T=clipart


RESUME=True
#
LOAD_NAME='./selected_max_3/res101/foodexclArtsmt10/globallocal_target_foodArtsmt10_eta_0.1_local_context_False_global_context_False_gamma_5_session_1_epoch_12_step_9999.pth'

DECAY_SETP=5
BCE=3
#--epochs

# !! train_domain_loss is necessary for training the domain align

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_weakly_backbone.py --cuda --net res101 \
   --dataset $DATASET --dataset_t $DATASET_T \
   --save_dir $2 \
   --nw $NW \
   --bce_alpha $BCE \
   --lr_decay_step=$DECAY_SETP \
   --weakly_type select_max \
   --checkpoint_interval 2000 \
   --r $RESUME --load_name $LOAD_NAME \

   #--train_region_wda_loss \
   #--train_img_wda_loss \
   #--epochs
   #--fixed_layer=4 \
   #--gc --lc \
