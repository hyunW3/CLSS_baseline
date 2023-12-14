#!/bin/bash
## 1117
# try to find out why MiB is not working
# to check boost lr effect 
cd ../z_exp_individual_cmd

# PLOP code
# ./train_voc_15-1_PLOP_2GPU.sh &
# ./train_voc_15-1_PLOP_2GPU_1.sh --set_deterministic 

# ./train_voc_15-1_PLOP_4GPU.sh
# after revision (remove bg classifier detach)
# ./train_voc_15-1_MiB_onlyIncremental.sh 
# ./train_voc_15-1_MiB.sh 

# 10-1
./train_voc_10-1_MiB.sh
wait_exec 90787 echo ""
wait_exec 60621 echo ""
wait_exec 61618 echo ""
alert_knock echo "all task is done"
./train_voc_10-1_PLOP.sh
./train_voc_10-1_DKD.sh