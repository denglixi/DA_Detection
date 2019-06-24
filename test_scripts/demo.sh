#!/bin/sh

GPU_ID=1

IMAGEDIR=../data/foggycity/JPEGImages/
MODELDIR=./models/
CFG_FILE=./cfgs/demo.yml

#webcam_num -2 show image online -1 saveimage

# load weight
SESSION=444
EPOCH=35
CHECKPOINT=14819


# basic set
#DATASET=foodEconomicBeeHoon
DATASET=foodAllmt10
NET=foodres50 #{foodres50, res101, vgg16}

WEB_NUM=-1

CUDA_VISIBLE_DEVICES=$GPU_ID python sources/demo.py --net $NET \
               --cfg $CFG_FILE \
               --dataset $DATASET \
               --image_dir $IMAGEDIR \
               --load_dir $MODELDIR \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --webcam_num $WEB_NUM \
               --cuda
