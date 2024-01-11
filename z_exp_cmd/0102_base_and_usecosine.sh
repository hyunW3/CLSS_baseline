#!/bin/bash

cd ../z_exp_individual_cmd

# naive finetune , no use cosine or use cosine
# ./train_voc_15-1_base.sh 0,1 &  
# ./train_voc_15-1_base.sh 2,3 --use_cosine
# sleep 10m;
# ./train_voc_15-1_base.sh 0,1 --OCFM &
# ./train_voc_15-1_base.sh 2,3 --OCFM --use_cosine
# ./train_voc_15-1_MiB.sh 0,1 --use_cosine
# alert_knock echo "0102_base_and_usecosine.sh finished"

# 0108 ver3
# ./train_voc_15-1_base.sh 0,1 --main_loss MBCE --use_cosine &  
# ./train_voc_15-1_base.sh 2,3 --main_loss CE --use_cosine
# sleep 10m;
# ./train_voc_15-1_base.sh 0,1 --main_loss MBCE --OCFM --use_cosine &
# ./train_voc_15-1_base.sh 2,3 --main_loss CE --OCFM --use_cosine
# ./train_voc_15-1_base.sh 0,1 --main_loss MBCE &
# ./train_voc_15-1_base.sh 2,3 --main_loss CE 

# 0109 
# ./train_voc_15-1_base.sh 0,1 --main_loss BCE --onlyBase &
# ./train_voc_15-1_base.sh 2,3 --main_loss BCE --use_cosine --onlyBase 

# wait_exec 112876 echo "waiting for 0,1 GPU task end"

# ./train_voc_15-1_base.sh 0,1 --main_loss MBCE --onlyBase &
# ./train_voc_15-1_base.sh 2,3 --main_loss MBCE --use_cosine --onlyBase 
