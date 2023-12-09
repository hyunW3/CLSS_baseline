#!/bin/bash
## 1117
# try to find out why MiB is not working
# to check boost lr effect 
cd ../z_exp_individual_cmd
# ./train_voc_15-1_MiB_onlyIncremental.sh
# ./train_voc_15-1_MiB_onlyIncremental.sh --boost_lr
# ./train_voc_15-1_MiB_onlyIncremental.sh --boost_lr --set_deterministic
# ./train_voc_15-1_MiB_onlyIncremental.sh --boost_lr --set_deterministic

# # ./train_voc_15-1_MiB.sh --set_deterministic # whole step success

# # 1119
# # effect of memory size
# ./train_voc_15-1_MiBwMemory.sh
# ./train_voc_15-1_MiBwMemory.sh --set_deterministic
# ./train_voc_15-1_MiBwMemory.sh --boost_lr
# ./train_voc_15-1_DKDwMemory.sh

# 1124 reproducity
./train_voc_15-1_MiB_unkd0.sh --kd 0
./train_voc_15-1_MiB_unkd0.sh --kd 1 --set_deterministic 
./train_voc_15-1_MiB_unkd0.sh --kd 5 --set_deterministic 
./train_voc_15-1_MiB_unkd0.sh --kd 2 --set_deterministic 
./train_voc_15-1_MiB_unkd0.sh --kd 0.1 --set_deterministic 
