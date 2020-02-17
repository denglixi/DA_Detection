#!/bin/sh

# test collected canteen
#DATASET=foodUTownmt10
DATASET=foodArtsmt10


# test excl canteen
#DATASET=foodexclArts
#DATASET=foodexclUTown

# NET=prefood #{res101, prefood, res101_local_unreversed}
NET=res101
CUDA_VISIBLE_DEVICES=$1 python ./app/test_backbone.py --cuda --net $NET --dataset $DATASET --load_name $2 \
    #--vis
    #--gc --lc \
    #--test_cache
    # --gc \
    # --nw 10 \
