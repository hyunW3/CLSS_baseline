#!/bin/bash

cd ../z_exp_individual_cmd

# naive finetune , no use cosine or use cosine
./train_voc_15-1_base.sh --onlyIncremental
./train_voc_15-1_base.sh --onlyIncremental --use_cosine

alert_knock echo "0102_base_and_usecosine.sh finished"
