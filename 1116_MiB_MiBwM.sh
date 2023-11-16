#!/bin/bash

./train_voc_15-1_MiB.sh &&
./train_voc_15-1_MiB.sh --set_deterministic &&
./train_voc_15-1_DKDwMemory.sh &&
./train_voc_15-1_MiBwMemory.sh &&
./train_voc_15-1_DKD.sh --set_deterministic
alert_knock echo "deterministic test"
./train_voc_15-1_MiB.sh --set_deterministic 
./train_voc_15-1_DKD.sh --set_deterministic
./train_voc_15-1_MiB.sh 
./train_voc_15-1_DKD.sh 

