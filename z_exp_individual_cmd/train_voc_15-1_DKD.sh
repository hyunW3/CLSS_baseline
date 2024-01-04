#!/bin/bash
cd ../
PORT='tcp://127.0.0.1:12345'
GPU=0,1,2,3
BS=8  # Total 32
SAVEDIR='saved_voc'

OPTION=$@ # --set_deterministic
TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-1'
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=0  # 100 for DKD-M

# check deterministic string exist in OPTION 
if [[ $OPTION == *"--set_deterministic"* ]]; then
    echo "deterministic version"
    NAME='DKD_deterministic'
else
    echo "non-deterministic version"
    NAME='DKD_non_deterministic'
fi
# check MEMORY_SIZE == 0 then add into NAME
if [ $MEMORY_SIZE -ne 0 ]; then
    NAME=${NAME}_M${MEMORY_SIZE}
fi


# check option --onlyIncremental
# and remove it from OPTION
onlyIncremental=false
if [[ $OPTION == *"--onlyIncremental"* ]]; then
    echo "=== onlyIncremental === "
    NAME=${NAME}
    OPTION=${OPTION/--onlyIncremental/}
    onlyIncremental=true
fi

# check onlyIncremental is True
if [ $onlyIncremental = false ]; then
    alert_knock python train_voc.py -c configs/config_voc_DKD.json \
    -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
    --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT} 
fi
alert_knock python train_voc.py -c configs/config_voc_DKD.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_DKD.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_DKD.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_DKD.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_DKD.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

alert_knock python eval_voc.py -d 0 --test \
    -r ${SAVEDIR}/${TASKSETTING}_${TASKNAME}_${NAME}/step_5/checkpoint-epoch${EPOCH}.pth # saved_voc/models/overlap_15-1_DKD/step_5/checkpoint-epoch60.pth


echo "voc 15-1 DKD done"