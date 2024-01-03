#!/bin/bash

cd ../z_exp_individual_cmd



./train_voc_10-1_PLOPat6.sh 2,3 --onlyIncremental # updated
./train_voc_15-1_PLOP.sh 2,3
./train_voc_15-1_MiB.sh 2,3


alert_knock echo "0103_train_voc_1.sh finished"