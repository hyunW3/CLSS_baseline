#!/bin/bash

cd ../z_exp_individual_cmd


./train_voc_5-3_PLOP_at3.sh 0,1 --onlyIncremental
./train_voc_10-1_MiB.sh 0,1 

alert_knock echo "0103_train_voc_0.sh finished"