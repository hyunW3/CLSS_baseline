#!/bin/bash

cd ../z_exp_individual_cmd


# ./train_voc_15-1_DKD.sh
# ./train_voc_10-1_DKD.sh
# ./train_voc_5-3_DKD.sh

# ./train_voc_15-1_MiB.sh 0,1 &
# ./train_voc_10-1_MiB.sh 2,3
# sleep 30m;

# ./train_voc_15-1_PLOP.sh 0,1 &
# ./train_voc_10-1_PLOP.sh 2,3 
# sleep 30m;
# ./train_voc_5-3_MiB.sh 0,1 &
# ./train_voc_15-1_PLOP.sh 2,3

# ./train_voc_5-3_MiB.sh 0,1 &
# ./train_voc_10-1_PLOP.sh 0,1 &
# ./train_voc_5-3_PLOP.sh 2,3
# ./train_voc_15-1_MiB.sh 0,1 &

# 240102
# ./train_voc_10-1_DKD_onlyLast.sh 

# ./train_voc_10-1_MiB_onlyLast.sh 0,1 & 

# TODO (OOM) - it is not working (messy)
# ./train_voc_5-3_PLOP.sh 0,1 --onlyIncremental & # 
# ./train_voc_10-1_PLOP.sh 2,3 --onlyIncremental # 텔레그램 O
# sleep 10m;
# ./train_voc_10-1_MiB.sh 0,1 &
# ./train_voc_15-1_PLOP.sh 2,3 # 이건 돌아감. -> 근데 결과가 너무 낮음
# ./train_voc_15-1_MiB.sh 2,3

# 0103
# ./train_voc_5-3_PLOP_at3.sh 0,1 --onlyIncremental
# ./train_voc_10-1_PLOPat6.sh 2,3 --onlyIncremental
# ./train_voc_10-1_MiB.sh 0,1 &
# ./train_voc_15-1_PLOP.sh 2,3
# ./train_voc_15-1_MiB.sh 2,3

# 0104 
./train_voc_15-1_DKD.sh


