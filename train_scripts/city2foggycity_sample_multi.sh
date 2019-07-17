#!/bin/sh

RESUME=true
LOAD_NAME='./models/vgg16_multiscale/cityscape/globallocal_target_foggy_cityscape_eta_0.1_local_context_False_global_context_False_gamma_5_session_1_epoch_2_step_9999.pth'

    #--r $RESUME --load_name $LOAD_NAME \
CUDA_VISIBLE_DEVICES=$1 python ./app/trainval_net_multiscale.py \
    --cuda --net vgg16_multiscale \
    --train_domain_loss \
    --dataset cityscape --dataset_t foggy_cityscape \
    --save_dir models \
    #--nw 5
    #--gc --lc \
