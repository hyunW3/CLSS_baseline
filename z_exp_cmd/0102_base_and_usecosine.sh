#!/bin/bash

cd ../z_exp_individual_cmd

# naive finetune , no use cosine or use cosine
./train_voc_15-1_base.sh 
./train_voc_15-1_base.sh --use_cosine
