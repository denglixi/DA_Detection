#!/bin/sh

NET=vgg16_weakly_sum
RESUME=true
#
LOAD_NAME='./models/vgg16_weakly_sum/cityscape/globallocal_target_foggy_cityscape_eta_0.1_local_context_True_global_context_True_gamma_5_session_1_epoch_9_step_9999.pth'

#--r RESUME --load_name $LOAD_NAME \
echo '--------training net-----------'
echo $NET

EPOCH=30

CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_global_local_weakly.py \
    --cuda --net $NET \
    --epochs $EPOCH \
    --r $RESUME --load_name $LOAD_NAME \
    --train_domain_loss \
    --dataset cityscape --dataset_t foggy_cityscape \
    --gc --lc --save_dir models \
    --nw 5
