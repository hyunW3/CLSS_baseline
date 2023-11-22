#!/bin/bash
cd ../z_exp_individual_cmd
# currnet running code (overlap_15-1_MiB_non_deterministic_test)
# : no bias correction for bg
# 1117 12:17 - init_novel_classifier
# no bias correction for bg 
# 1117 23:06 - init_novel_classifier won't make any difference (no effect on training trouble)

# TODO bias[0] for bg init
# ./train_voc_15-1_MiB.sh  # 
# ./train_voc_15-1_DKD.sh 

# check duplicated results as before 
# alert_knock echo "deterministic test"
# ./train_voc_15-1_MiB.sh --set_deterministic 
# ./train_voc_15-1_DKD.sh --set_deterministic
# ./train_voc_15-1_MiB.sh --set_deterministic 
# ./train_voc_15-1_DKD.sh --set_deterministic

# check duplicated results as before
# ./train_voc_15-1_MiB.sh 

# ./train_voc_15-1_DKDwMemory.sh &&
# ./train_voc_15-1_MiBwMemory.sh 

./train_voc_5-3.sh