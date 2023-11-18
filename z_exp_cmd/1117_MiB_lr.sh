#!/bin/bash
# try to find out why MiB is not working
# to check boost lr effect 
cd ../
./train_voc_15-1_MiB.sh 
./train_voc_15-1_MiB.sh --boost_lr
./train_voc_15-1_MiB.sh --boost_lr --set_deterministic
./train_voc_15-1_MiB.sh --boost_lr --set_deterministic

./train_voc_15-1_MiB.sh --set_deterministic

