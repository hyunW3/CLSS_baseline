#!/bin/bash
cd ../
PORT='tcp://127.0.0.1:12345'
GPU=0,1,2,3
BS=6  # Total 24
SAVEDIR='saved_voc'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-1'
EPOCH=30
INIT_LR=0.01
LR=0.001
OPTION=$@ # --set_deterministic
# INIT_POSWEIGHT=2
MEMORY_SIZE=0  # 100 for DKD-M
# parse option --kd 5
if [[ $OPTION == *"--kd"* ]]; then
    echo "kd version"
    KD=`echo $OPTION | sed -n 's/.*--kd \([^ ]*\).*/\1/p'`
    OPTION=${OPTION/--kd $KD/}
    echo "KD: $KD"
    echo "OPTION: $OPTION"
    NAME='MiB_kd'${KD}
else
    echo "unkd version"
    NAME='MiB_unkd'
fi

# check deterministic string exist in OPTION 
if [[ $OPTION == *"--set_deterministic"* ]]; then
    echo "deterministic version"
    NAME=$NAME'_MiB_deterministic'
else
    echo "non-deterministic version"
    NAME=$NAME'_MiB_non_deterministic'
fi
if [[ $OPTION == *"--boost_lr"* ]]; then
    echo "=== boost lr === "
    NAME=${NAME}_boost_lr
fi
# check MEMORY_SIZE == 0 then add into NAME
if [ $MEMORY_SIZE -ne 0 ]; then
    NAME=${NAME}_M${MEMORY_SIZE}
fi
# NAME=${NAME}_$OPTION
echo "NAME: $NAME"

alert_knock python train_voc.py -c configs/config_voc_MiB_unkd0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} && 

alert_knock python train_voc.py -c configs/config_voc_MiB_unkd0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB_unkd0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB_unkd0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB_unkd0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB_unkd0.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_voc.py -d 0 \
    -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_5/checkpoint-epoch${EPOCH}.pth # saved_voc/models/overlap_15-1_DKD/step_5/checkpoint-epoch60.pth
# python eval_voc.py -d 0 -r saved_voc/models/overlap_15-1_DKD/step_5/checkpoint-epoch60.pth