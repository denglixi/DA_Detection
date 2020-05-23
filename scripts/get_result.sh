#!/bin/sh
# basic set 
#DATASET=foodYIHmt10
#DATASET=foodexclYIHmt10_testYIH
#DATASET=food$Train$mt$Test

#DATASET=foodexclYIHmt10_testYIHfew1
#DATASET=foodexclYIH_fineYIHfew5_testYIHfew5
#DATASET=foodexclYIH_testYIHmt10
#DATASET=foodexclYIHmt10_testYIHfew1

#DATASET=foodexclArtsmt10
#DATASET=foodexclArtsmt10_testArtsfew1

#DATASET=foodexclUTownmt10_testUTownfew1
#DATASET=foodexclUTownmt10_fineUTownfew5_testUTownfew5

#DATASET=foodexclSciencemt10_testScience

#DATASET=foodAllmt10

#DATASET=foodAllmt10
# test collected canteen
DATASET=foodUTownmt10

# test excl canteen
# DATASET=foodexclArts

# whether save all detection results in images
SAVE_FOR_VIS= # blank for false

#FasterModel='/storage/lxdeng/DA_Detection/output/faster_foodres50/food_Arts_innermt10test/faster_rcnn_5_14_13349/detections.pkl'
#LocalModel='/storage/lxdeng/DA_Detection/output/foodres50/food_Arts_innermt10test/local_target_foodArts_eta_0.1_efocal_False_local_context_True_gamma_5_session_1_epoch_14_step_9999.pth/detections.pkl'
#GlobalModel='/storage/lxdeng/DA_Detection/output/foodres50/food_Arts_innermt10test/global_target_foodArts_eta_0.1_efocal_False_global_context_True_gamma_5_session_1_epoch_14_step_9999.pth/detections.pkl'
#GlobalLocalModel='/storage/lxdeng/DA_Detection/output/DA_g+l_foodres50/food_Arts_innermt10test/globallocal_target_foodArts_eta_0.1_local_context_True_global_context_True_gamma_5_session_1_epoch_14_step_9999.pth/detections.pkl'

baseline='/Users/lixi/Works/DA_Detection/UTown/base/detections.pkl'
MTW='/Users/lixi/Works/DA_Detection/UTown/mt/detections.pkl'

CUDA_VISIBLE_DEVICES=$GPU_ID python ./app/draw_det_results.py \
                   --dataset $DATASET \
                   --save_for_vis=$SAVE_FOR_VIS \
                   --model_result_paths $baseline $MTW
