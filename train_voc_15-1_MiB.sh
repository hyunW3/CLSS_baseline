#!/bin/bash

PORT='tcp://127.0.0.1:12345'
GPU=0,1,2,3
BS=6  # Total 24
SAVEDIR='saved_voc'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-1'
EPOCH=30
INIT_LR=0.01
LR=0.001
# INIT_POSWEIGHT=2
MEMORY_SIZE=0  # 100 for DKD-M

NAME='MiB'
# if `find ./saved_voc/models/overlap_15-1_MiB/ -name "*.pth"` ; then
#     echo "Found saved model."
#     echo "==========="
#     echo `find ./saved_voc/models/overlap_15-1_MiB/ -name "*.pth"`
#     echo "==========="
#     echo "Do you want to remove whole folder? [y/n]"
#     read answer
#     if [ $answer = "y" ] ; then
#         rm -r saved_voc/models/overlap_15-1_MiB/*
#         rm -r saved_voc/log/overlap_15-1_MiB/*
#     else 
#         echo "Continue."
#     fi
# else
#     echo "No saved model found."
#     rm -r saved_voc/models/overlap_15-1_MiB/*
#     rm -r saved_voc/log/overlap_15-1_MiB/*
# fi

alert_knock python train_voc.py -c configs/config_voc_MiB_seed0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} # --pos_weight ${INIT_POSWEIGHT}

alert_knock python train_voc.py -c configs/config_voc_MiB_seed0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

alert_knock python train_voc.py -c configs/config_voc_MiB_seed0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

alert_knock python train_voc.py -c configs/config_voc_MiB_seed0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

alert_knock python train_voc.py -c configs/config_voc_MiB_seed0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

alert_knock python train_voc.py -c configs/config_voc_MiB_seed0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_voc.py -d 0 -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_5/checkpoint-epoch${EPOCH}.pth # saved_voc/models/overlap_15-1_DKD/step_5/checkpoint-epoch60.pth
# python eval_voc.py -d 0 -r saved_voc/models/overlap_15-1_DKD/step_5/checkpoint-epoch60.pth