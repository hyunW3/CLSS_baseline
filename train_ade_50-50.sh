#!/bin/bash

GPU=0,1,2,3
BS=6  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='50-50'
EPOCH=100
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=0 # 300 for Ours-M

NAME='DKD'
if `find ./saved_ade/ -name "*.pth"` ; then
    echo "Found saved model. Do you want to remove it? [y/n]"
    read answer
    if [ $answer = "y" ] ; then
        rm -r saved_ade/*
    else 
        echo "Exit."
        exit 0
    fi
else
    echo "No saved model found."
    rm -r saved_ade/*
fi

alert_knock python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

alert_knock python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

alert_knock python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_ade.py -d 0 -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_2/checkpoint-epoch${EPOCH}.pth
