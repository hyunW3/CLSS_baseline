#!/bin/bash

cd ../z_exp_individual_cmd


# ./train_voc_10-1_MiB.sh 0,1 
# ./train_voc_5-3_PLOP.sh 0,1
./train_voc_15-1_DKD.sh 0,1

alert_knock echo "0103_train_voc_0.sh finished"