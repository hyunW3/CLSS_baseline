#!/bin/bash
cd ../
PORT='tcp://127.0.0.1:12345'
GPU=$1
BS=12  # Total 24
SAVEDIR='saved_voc'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='10-1'
EPOCH=30
INIT_LR=0.01
LR=0.001

MEMORY_SIZE=0  # 100 for MiB-M

NAME='MiB'
# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 6 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 7 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 8 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

# python train_voc.py -c configs/config_voc_MiB.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 9 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 10 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_voc.py -d 0 --test -r ${SAVEDIR}/${TASKSETTING}_${TASKNAME}_${NAME}/step_10/checkpoint-epoch${EPOCH}.pth
